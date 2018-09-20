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
  diagCov->set(1);
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
  _dispose_object(diagCov);
  _dispose_object(avgNois);

 for(auto& ptr: popNoiseVectors) _dispose_object(ptr);
 for(auto& ptr: generators) _dispose_object(ptr);
}

void CMA_Optimizer::initializeGeneration() const {
  const nnReal* const S = diagCov->params;
  const nnReal* const M = weights->params;
  const nnReal _eta = bAnnealLearnRate? annealRate(eta,nStep,epsAnneal) : eta;

  #pragma omp parallel for schedule(static)
  for(Uint i=1; i<pop_size; i++) {
    Saru& gen = * generators[i];
    nnReal* const Y = popNoiseVectors[i]->params;
    nnReal* const X = sampled_weights[i]->params;
    for(Uint w=0; w<pDim; w++) {
      #if 1
      Y[w] = gen.f_mean0_var1();
      X[w] = M[w] + _eta * Y[w] * S[w];
      #else
      Y[w] = gen.f_mean0_var1() * S[w];
      X[w] = M[w] + _eta * Y[w];
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

  //cout << "pre:"<< print(inds) << endl;
  if(true) // add weight penalization to prevent drift
  {
    vector<nnReal> Wnorms (pop_size, 0);
    for(Uint i=0; i<pop_size; i++) {
      const nnReal* const X = sampled_weights[i]->params;
      nnReal sum2 = 0;
      #pragma omp parallel for simd schedule(static) reduction(+ : sum2)
      for(Uint w=0; w<pDim; w++) sum2 += X[w] * X[w];
      Wnorms[i] = sum2; // std::sqrt(sum2);
    }

    Real E2W2 = 0, EW2 = 0, E2L = 0, EL = 0; Uint cnt = 0;
    for(Uint i=0; i<pop_size; i++) {
      if(popWeights[i] <= 0) continue;
      cnt ++;
      const Uint k = inds[i];
      E2W2 += Wnorms[k] * Wnorms[k]; EW2 += Wnorms[k];
      E2L  += losses[k] * losses[k]; EL  += losses[k];
    }
    const Real nrm = 1.0/cnt;
    const Real varW2 = nrm * (E2W2 - nrm * EW2 * EW2);
    const Real varL  = nrm * (E2L  - nrm * EL  * EL );
    const Real meanW2 = nrm * EW2, meanL = nrm * EL;
    const Real nrmL = 1/std::sqrt(varL), nrmW2 = 0.1/std::sqrt(varW2);
    for(Uint i=0; i<pop_size; i++)
     losses[i] = nrmL * (losses[i]-meanL) + nrmW2 * (Wnorms[i] - meanW2);
    std::sort(inds.begin(), inds.end(), // is i1 before i2
       [&] (const Uint i1, const Uint i2) { return losses[i1] < losses[i2]; } );
  }
  //cout << "pst:"<< print(inds) << endl;

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
    const nnReal* const S = diagCov->params;
    nnReal * const M = weights->params;
    nnReal * const B = momNois->params;
    nnReal * const A = avgNois->params;
    #pragma omp for simd schedule(static) aligned(B,A,Y,M,X,S : VEC_WIDTH)
    for(Uint w=0; w<pDim; w++) {
      B[w] += wC * std::pow( Y[w] * S[w], 2);
      M[w] += wZ * X[w];
      #if 0
      A[w] += wZ * Y[w];
      #else
      A[w] += wZ * Y[w] * S[w];
      #endif
    }
  }

  const nnReal * const B = momNois->params;
  const nnReal * const A = avgNois->params;
  nnReal * const C = pathCov->params;
  nnReal * const S = diagCov->params;
  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  const nnReal alpha = 1 - 2*c1cov - sumW*mu_eff*c1cov;
  #pragma omp parallel for simd schedule(static) aligned(B,S,C,A : VEC_WIDTH)
  for(Uint w=0; w<pDim; w++) {
    C[w] = (1-c_sig)*C[w] + updSigP*A[w];
    S[w] = std::sqrt( alpha*S[w]*S[w] + 2*c1cov*C[w]*C[w] + mu_eff*c1cov*B[w] );
  }

  initializeGeneration();
}

void CMA_Optimizer::save(const string fname, const bool backup) {
  weights->save(fname+"_weights");
  pathCov->save(fname+"_pathCov");
  diagCov->save(fname+"_diagCov");

  if(backup) {
    ostringstream ss; ss << std::setw(9) << std::setfill('0') << nStep;
    weights->save(fname+"_"+ss.str()+"_weights");
    pathCov->save(fname+"_"+ss.str()+"_pathCov");
    diagCov->save(fname+"_"+ss.str()+"_diagCov");
  }
}
int CMA_Optimizer::restart(const string fname) {
  pathCov->restart(fname+"_pathCov");
  diagCov->restart(fname+"_diagCov");
  return weights->restart(fname+"_weights");
}
