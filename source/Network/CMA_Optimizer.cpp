//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Optimizer.h"
#include "saruprng.h"

CMA_Optimizer::CMA_Optimizer(Settings&S, const Parameters*const W,
  const Parameters*const W_TGT): mastersComm(S.mastersComm),
  learn_size(S.learner_size), pop_size(S.ESpopSize), weights(W),
  tgt_weights(W_TGT), sigma_init(S.learnrate) {
    diagCov->set(1);
    std::vector<unsigned long> seed(3*pop_size) ;
    std::generate(seed.begin(), seed.end(), [&](){return S.generators[0]();});
    MPI_Bcast(seed.data(), 3*pop_size, MPI_UNSIGNED_LONG, 0, mastersComm);
    for(Uint i=0; i<pop_size; i++)
      generators[i] = new Saru(seed[3*i +0], seed[3*i +1], seed[3*i +2]);
  }

virtual CMA_Optimizer::~CMA_Optimizer() {
 _dispose_object(pathSig);
 _dispose_object(pathCov);
 _dispose_object(diagCov);
 for(auto& ptr: sampled_weights) _dispose_object(ptr);
 for(auto& ptr: popNoiseVectors) _dispose_object(ptr);
 for(auto& ptr: generators) _dispose_object(ptr);
}

void CMA_Optimizer::initializeGeneration() const {
  #pragma omp parallel for schedule(static)
  for(Uint i=0; i<pop_size; i++) {
    Saru& gen = generators[i];
    nnReal* const Z = popNoiseVectors[i]->params;
    nnReal* const X = sampled_weights[i]->params;
    const nnReal* const S = diagCov->params;
    const nnReal* const M = weights->params;
    for(Uint w=0; w<pDim; w++) {
      Z[w] = gen.f_mean0_var1();
      X[w] = M[w] + sigma * S[w] * Z[w];
    }
  }
}

void CMA_Optimizer::prepare_update(const vector<Real> L) {
  if(L.size() not_eq pop_size) die("");
  losses = L;
  if (learn_size > 1) //add up losses across master ranks
    MPI_Iallreduce(MPI_IN_PLACE, losses.data(), pop_size, MPI_VALUE_TYPE, MPI_SUM, mastersComm, &paramRequest);
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
  std::sort(inds.begin(), inds.end(),
              [&](Uint i1, Uint i2) { return losses[i1] < losses[i2]; } );

  nnReal * const M = weights->params;
  nnReal * const S = diagCov->params;
  nnReal * const P = pathCov->params;
  nnReal * const C = pathSig->params;
  nnReal * const A = avgNois->params;
  avgNois->clear(); weights->clear();

  #pragma omp parallel
  for(Uint i=0; i<pop_size; i++) {
    const Uint k = inds[i];
    const nnReal ww = popWeights[i]
    const nnReal* const Z = popNoiseVectors[k]->params;
    const nnReal* const X = sampled_weights[k]->params;
    #pragma omp for simd schedule(static) aligned(A,Z,M,X : VEC_WIDTH)
    for(Uint w=0; w<pDim; w++) { A[w] += Z[w] * ww; M[w] += X[w] * ww; }
  }

  nnReal sumPP = 0, sumSS = 0;
  #pragma omp parallel for simd schedule(static) reduction(+ : sumSS)
  for(Uint w=0; w<pDim; w++) {
    C[w] = (1-c_sig)*C[w] + updSigP*A[w];
    sumSS += C[w] * C[w];
  }

  const nnReal updNormSigm = std::sqrt( sumSS / pDim );
  sigma *= std::exp( updSigm * ( updNormSigm - 1 ) );
  const nnReal hsig = updNormSigm < ((1.4 + 2./pDim) * std::sqrt(1-anneal));
  anneal *= std::pow( 1 - c_sig, 2 );
  if(anneal < 2e-16) anneal = 0;

  #pragma omp parallel for simd schedule(static) reduction(+ : sumPP)
  for(Uint w=0; w<pDim; w++) {
    P[w] = (1-c1cov) * P[w] + hsig*updPath * A[w];
    sumPP += P[w] * P[w];
  }

  const nnReal covBeta = ( std::sqrt(1 + sumPP*c2cov/(1-c2cov)) - 1 )/sumPP;
  #pragma omp parallel for simd schedule(static)
  for(Uint w=0; w<pDim; w++) S[w] = covAlph*(S[w] + covBeta*S[w]*P[w]*P[w]);
}

void CMA_Optimizer::save(const string fname, const bool backup) {
  weights->save(fname+"_weights");
  pathSig->save(fname+"_pathSig");
  pathCov->save(fname+"_pathCov");
  diagCov->save(fname+"_diagCov");

  if(backup) {
    ostringstream ss; ss << std::setw(9) << std::setfill('0') << nStep;
    weights->save(fname+"_"+ss.str()+"_weights");
    pathSig->save(fname+"_"+ss.str()+"_pathSig");
    pathCov->save(fname+"_"+ss.str()+"_pathCov");
    diagCov->save(fname+"_"+ss.str()+"_diagCov");
  }
}
int CMA_Optimizer::restart(const string fname) {
  pathSig->restart(fname+"_pathSig");
  pathCov->restart(fname+"_pathCov");
  diagCov->restart(fname+"_diagCov");
  return weights->restart(fname+"_weights");
}
