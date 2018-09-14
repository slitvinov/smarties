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
  cout << mu_eff << endl;
  diagCov->set(1);
  pathSig->set(1);
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
  const nnReal* const S = diagCov->params;
  const nnReal* const M = weights->params;

  #if 0
  #pragma omp parallel for schedule(static)
  for(Uint i=1; i<pop_size; i++) {
    Saru& gen = * generators[i];
    nnReal* const Z = popNoiseVectors[i]->params;
    nnReal* const X = sampled_weights[i]->params;
    for(Uint w=0; w<pDim; w++) {
      Z[w] = gen.f_mean0_var1();
      X[w] = M[w] + sigma * S[w] * Z[w];
    }
  }
  #else
  #pragma omp parallel for schedule(static)
  for(Uint i=1; i<pop_size; i+=2) {
    Saru& gen = * generators[i];
    nnReal* const Z1 = popNoiseVectors[i]->params;
    nnReal* const Z2 = popNoiseVectors[i+1]->params;
    nnReal* const X1 = sampled_weights[i]->params;
    nnReal* const X2 = sampled_weights[i+1]->params;
    for(Uint w=0; w<pDim; w++) {
      Z1[w] = gen.f_mean0_var1(); Z2[w] = -Z1[w];
      X1[w] = M[w] + sigma * S[w] * Z1[w];
      X2[w] = M[w] + sigma * S[w] * Z2[w];
    }
  }
  #endif
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
    //#pragma omp master
    //cout<<losses[k]<<" "<<popWeights[i]<<endl;
    const nnReal wZ = popWeights[i];
    const nnReal wM = k ? _eta*popWeights[i] : _eta*popWeights[i] + (1-_eta);
    const nnReal* const Z = popNoiseVectors[k]->params;
    const nnReal* const X = sampled_weights[k]->params;
    if(wM <= 0) continue;
    #pragma omp for simd schedule(static) aligned(A,Z,M,X : VEC_WIDTH)
    for(Uint w=0; w<pDim; w++) { A[w] += Z[w] * wZ; M[w] += X[w] * wM; }
  }

  nnReal * const C = pathSig->params;
  nnReal * const S = diagCov->params;

  Real sumSS = 0, sumCC = 0;
  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  #pragma omp parallel for simd schedule(static) reduction(+ : sumSS, sumCC)
  for(Uint w=0; w<pDim; w++) {
    C[w] = (1-c_sig)*C[w] + updSigP*A[w];
    sumCC += C[w] * C[w]; sumSS += S[w];
  }

  //const nnReal updSigm = c_sig / ( 1 + c_sig );
  //const nnReal updNormSigm = std::sqrt( sumCC / pDim ) / (1 - 0.25/pDim);
  //sigma *= std::exp( updSigm * ( updNormSigm - 1 ) );
  //sigma = std::min( sigma, (nnReal) eta );
  const nnReal alph = std::sqrt( 1 - c1cov ) * pDim / sumSS;
  const nnReal beta = ( std::sqrt( 1 + sumCC*c1cov/(1-c1cov) ) - 1 )/sumCC;
  #pragma omp parallel for simd schedule(static)
  for(Uint w=0; w<pDim; w++) S[w] = alph*S[w]*(1 + beta*C[w]*C[w]);

  if((nStep%1000)==0) cout<<sigma<<" "<<sumCC<<" "<<sumSS<<endl;
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
