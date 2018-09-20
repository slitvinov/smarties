//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "CMA_Optimizer.h"
#include "saruprng.h"
#include <algorithm>

CMA_Optimizer::CMA_Optimizer(Settings&S, const Parameters*const W,
  const Parameters*const WT, const vector<Parameters*>&G) : Optimizer(S, W, WT),
  sampled_weights(G) {
  diagPos->set(1);
  diagNeg->set(1);
  std::vector<unsigned long> seed(3*pop_size) ;
  std::generate(seed.begin(), seed.end(), [&](){return S.generators[0]();});
  MPI_Bcast(seed.data(), 3*pop_size, MPI_UNSIGNED_LONG, 0, mastersComm);
  generators.resize(pop_size, nullptr);
  for(Uint i=0; i<pop_size; i++)
    generators[i] = new Saru(seed[3*i +0], seed[3*i +1], seed[3*i +2]);
  initializeGeneration();
}

CMA_Optimizer::~CMA_Optimizer() {
  _dispose_object(momNois);
  _dispose_object(diagPos);
  _dispose_object(diagNeg);

 for(auto& ptr: popNoiseVectors) _dispose_object(ptr);
 for(auto& ptr: generators) _dispose_object(ptr);
}

void CMA_Optimizer::initializeGeneration() const {
  const nnReal* const S = diagPos->params;
  const nnReal* const Z = diagNeg->params;
  const nnReal* const M = weights->params;
  const nnReal _eta = bAnnealLearnRate? annealRate(eta,nStep,epsAnneal) : eta;

  #pragma omp parallel for schedule(static)
  for(Uint i=1; i<pop_size; i++) {
    Saru& gen = * generators[i];
    nnReal* const Y = popNoiseVectors[i]->params;
    nnReal* const X = sampled_weights[i]->params;
    for(Uint w=0; w<pDim; w++) {
      #if 1
        const nnReal eps = gen.f_mean0_var1();
        Y[w] = eps*(eps>0)*S[w] + eps*(eps<0)*Z[w];
        X[w] = M[w] + _eta * Y[w];
      #else
        Y[w] = gen.f_mean0_var1();
        X[w] = M[w] + _eta * ( Y[w]*(Y[w]>0)*S[w] + Y[w]*(Y[w]<0)*Z[w] );
      #endif
    }
  }
}

void CMA_Optimizer::prepare_update(const int BS, const Rvec&L) {
  assert(L.size() == pop_size);
  losses = L;
  if (learn_size > 1) //add up losses across master ranks
    MPI_Iallreduce(MPI_IN_PLACE, losses.data(), pop_size, MPI_VALUE_TYPE,
                   MPI_SUM, mastersComm, &paramRequest);
  nStep++;
}

void CMA_Optimizer::apply_update()
{
  if(nStep == 0) die("nStep == 0");
  if(learn_size > 1) {
    if(paramRequest == MPI_REQUEST_NULL) die("Did not start reduction");
    MPI_Wait(&paramRequest, MPI_STATUS_IGNORE);
  }

  std::vector<Uint> inds(pop_size,0);
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), // is i1 before i2
       [&] (const Uint i1, const Uint i2) { return losses[i1] < losses[i2]; } );

  nnReal * const M = weights->params;
  nnReal * const B = momNois->params;
  nnReal * const A = avgNois->params;
  sampled_weights[0]->copy(weights); // first backup mean weights
  popNoiseVectors[0]->clear();       // sample 0 is always mean W, no noise
  weights->clear(); // prepare for reduction
  momNois->clear(); avgNois->clear();

  #pragma omp parallel
  for(Uint i=0; i<pop_size; i++) {
    const Uint k = inds[i];
    const nnReal wZ = std::max(popWeights[i], (nnReal) 0), wC = popWeights[i];
    const nnReal* const Y = popNoiseVectors[k]->params;
    const nnReal* const X = sampled_weights[k]->params;
    #pragma omp for simd schedule(static) aligned(B,Y,M,X : VEC_WIDTH)
    for(Uint w=0; w<pDim; w++) {
      B[w] += wC * Y[w] * std::fabs(Y[w]);
      M[w] += wZ * X[w];
      A[w] += wC * Z[w];
    }
  }

  nnReal * const C = pathCov->params;
  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  #pragma omp parallel for simd schedule(static) aligned(C,A : VEC_WIDTH)
  for(Uint w=0; w<pDim; w++)  C[w] = (1-c_sig)*C[w] + updSigP*A[w];

  nnReal * const S = diagPos->params;
  nnReal * const Z = diagNeg->params;
  static constexpr nnReal eps = 1e-2;
  #pragma omp parallel for simd schedule(static) aligned(B,S,Z : VEC_WIDTH)
  for(Uint w=0; w<pDim; w++) {
    //S[w] = std::sqrt( std::max( S[w]*S[w] + c1cov*B[w], eps ) );
    //Z[w] = std::sqrt( std::max( Z[w]*Z[w] - c1cov*B[w], eps ) );
    S[w] = std::sqrt(std::max((1-c1cov)*S[w]*S[w] +c1cov*(1+B[w]), eps));
    Z[w] = std::sqrt(std::max((1-c1cov)*Z[w]*Z[w] +c1cov*(1-B[w]), eps));
  }
  initializeGeneration();
}

void CMA_Optimizer::save(const string fname, const bool backup) {
  weights->save(fname+"_weights");
  diagPos->save(fname+"_diagPos");
  diagNeg->save(fname+"_diagNeg");

  if(backup) {
    ostringstream ss; ss << std::setw(9) << std::setfill('0') << nStep;
    weights->save(fname+"_"+ss.str()+"_weights");
    diagPos->save(fname+"_"+ss.str()+"_diagPos");
    diagNeg->save(fname+"_"+ss.str()+"_diagNeg");
  }
}
int CMA_Optimizer::restart(const string fname) {
  diagPos->restart(fname+"_diagPos");
  diagNeg->restart(fname+"_diagNeg");
  return weights->restart(fname+"_weights");
}
