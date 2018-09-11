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
  sampled_weights(G)
{
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
 _dispose_object(pathSig);
 _dispose_object(pathCov);
 _dispose_object(diagCov);
 _dispose_object(avgNois);

 for(auto& ptr: popNoiseVectors) _dispose_object(ptr);
 for(auto& ptr: generators) _dispose_object(ptr);
}

void CMA_Optimizer::initializeGeneration() const {
  #pragma omp parallel for schedule(static)
  for(Uint i=1; i<pop_size; i++) {
    Saru& gen = * generators[i];
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

void CMA_Optimizer::prepare_update(const int BS, const vector<Rvec>&L) {
  losses = Rvec(pop_size, 0);
  for(Uint j=0; j<L.size(); j++) {
    assert(L[j].size() == pop_size);
    for(Uint i=0; i<pop_size; i++) losses[i] += L[j][i];
  }
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

  const nnReal _eta = bAnnealLearnRate? annealRate(eta,nStep,epsAnneal) : eta;

  std::vector<Uint> inds(pop_size,0);
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), // is i1 before i2
       [&] (const Uint i1, const Uint i2) { return losses[i1] < losses[i2]; } );

  nnReal * const M = weights->params;
  nnReal * const A = avgNois->params;
  sampled_weights[0]->copy(weights); // first backup mean weights
  popNoiseVectors[0]->clear();       // sample 0 is always mean W, no noise
  avgNois->clear(); weights->clear(); // prepare for reduction these fields

  #pragma omp parallel
  for(Uint i=0; i<pop_size; i++) {
    const Uint k = inds[i];
    const nnReal wZ = popWeights[i];
    const nnReal wM = k ? _eta*popWeights[i] : _eta*popWeights[i] + (1-_eta);
    const nnReal* const Z = popNoiseVectors[k]->params;
    const nnReal* const X = sampled_weights[k]->params;
    if(wM <= 0) continue;
    #pragma omp for simd schedule(static) aligned(A,Z,M,X : VEC_WIDTH)
    for(Uint w=0; w<pDim; w++) { A[w] += Z[w] * wZ; M[w] += X[w] * wM; }
  }

  nnReal * const C = pathSig->params;
  nnReal sumPP = 0, sumSS = 0;
  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  #pragma omp parallel for simd schedule(static) reduction(+ : sumSS)
  for(Uint w=0; w<pDim; w++) {
    C[w] = (1-c_sig)*C[w] + updSigP*A[w];
    sumSS += C[w] * C[w];
  }

  const nnReal updSigm = c_sig / ( 1 + c_sig );
  const nnReal updNormSigm = std::sqrt( sumSS / pDim );
  sigma *= std::exp( updSigm * ( updNormSigm - 1 ) );
  sigma = std::min( sigma, (nnReal) std::sqrt(eta) );
  //cout << "sigma: "<<sigma << endl;
  //const nnReal hsig = updNormSigm < ((1.4 + 2./pDim) * std::sqrt(1-anneal));
  //const nnReal hsig = updNormSigm < 1.5 * std::sqrt(1-anneal);
  const nnReal hsig = 1;
  //if (hsig<.5) cout << "triggered hsig==0" << endl;
  anneal *= std::pow( 1 - c_sig, 2 );
  if(anneal < 2e-16) anneal = 0;

  nnReal * const P = pathCov->params;
  const nnReal updPath = std::sqrt(cpath * (2-cpath) * mu_eff);
  #pragma omp parallel for simd schedule(static) reduction(+ : sumPP)
  for(Uint w=0; w<pDim; w++) {
    P[w] = (1-cpath) * P[w] + hsig*updPath * A[w];
    sumPP += P[w] * P[w];
  }

  nnReal * const S = diagCov->params;
  const nnReal covAlph = 1 - c1cov*(1 - (1-hsig) * cpath * (2-cpath));
  const nnReal sqrAlph = std::sqrt( covAlph );
  const nnReal covBeta = ( std::sqrt( 1 + sumPP*c1cov/covAlph ) - 1 )/sumPP;
  #pragma omp parallel for simd schedule(static)
  for(Uint w=0; w<pDim; w++) {
    S[w] = sqrAlph*S[w]*(1 + covBeta*P[w]*P[w]);
    S[w] = std::min((nnReal)1, S[w]);
  }

  initializeGeneration();
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
