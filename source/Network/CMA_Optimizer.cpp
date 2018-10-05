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

CMA_Optimizer::CMA_Optimizer(const Settings&S, const Parameters*const W,
  const Parameters*const WT, const vector<Parameters*>&G) : Optimizer(S, W, WT),
  sampled_weights(G) {
  diagCov->set(1);
  pathCov->set(0);
  pathDif->set(0);
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
  _dispose_object(avgNois);
  _dispose_object(negNois);

  _dispose_object(pathCov);
  _dispose_object(pathDif);
  _dispose_object(diagCov);

 for(auto& ptr: popNoiseVectors) _dispose_object(ptr);
 for(auto& ptr: generators) _dispose_object(ptr);
}

void CMA_Optimizer::initializeGeneration() const {
  const nnReal* const S = diagCov->params;
  const nnReal* const M = weights->params;
  //const nnReal* const D = pathDif->params;
  const nnReal _eta = bAnnealLearnRate? annealRate(eta,nStep,epsAnneal) : eta;
  #pragma omp parallel for schedule(static)
  for(Uint i=1; i<pop_size; i++) {
    Saru& gen = * generators[i];
    nnReal* const Y = popNoiseVectors[i]->params;
    nnReal* const X = sampled_weights[i]->params;
    for(Uint w=0; w<pDim; w++) {
      Y[w] = gen.f_mean0_var1() * S[w];
      X[w] = M[w] + _eta * Y[w]; //+ _eta*1e-2*D[w];
    }
  }
}

void CMA_Optimizer::prepare_update(const Rvec&L) {
  assert(L.size() == pop_size);
  losses = L;
  if (learn_size > 1) { //add up losses across master ranks
    MPI(Iallreduce, MPI_IN_PLACE, losses.data(), pop_size, MPI_VALUE_TYPE,
                   MPI_SUM, mastersComm, &paramRequest);
  }
  nStep++;
}

void CMA_Optimizer::apply_update()
{
  if(nStep == 0) die("nStep == 0");
  if(learn_size > 1) {
    if(paramRequest == MPI_REQUEST_NULL) die("Did not start reduction");
    MPI(Wait, &paramRequest, MPI_STATUS_IGNORE);
  }

  std::vector<Uint> inds(pop_size,0);
  std::iota(inds.begin(), inds.end(), 0);
  std::sort(inds.begin(), inds.end(), // is i1 before i2
       [&] (const Uint i1, const Uint i2) { return losses[i1] < losses[i2]; } );

  //cout << "pre:"<< print(inds) << endl;
  if(0) // add weight penalization to prevent drift
  {
    vector<nnReal> Wnorms(pop_size, 0);
    for(Uint i=0; i < pop_size / 2; i++) {
      const nnReal* const X = sampled_weights[inds[i]]->params;
      nnReal sum2 = 0;
      #pragma omp parallel for simd schedule(static) reduction(+ : sum2)
      for(Uint w=0; w<pDim; w++) sum2 += X[w] * X[w]; // std::fabs(X[w]) for L1
      Wnorms[inds[i]] = sum2;
    }
    {
      Real E2L = 0, EL = 0;
      for(Uint i=0; i<pop_size; i++){ E2L+=losses[i]*losses[i]; EL+=losses[i]; }
      const Real nrm = 1.0/pop_size;
      const Real precL = 1/std::sqrt(nrm*(E2L-nrm*EL*EL)), meanL = nrm*EL;
      for(Uint i=0; i < pop_size/2; i++)
        losses[inds[i]] = (losses[inds[i]]-meanL)*precL + 0.1*Wnorms[inds[i]];
    }
    const auto oldSrt = inds;
    std::sort(&(inds[0]), &(inds[pop_size/2]), // is i1 before i2
       [&] (const Uint i1, const Uint i2) { return losses[i1] < losses[i2]; } );
    for(Uint i=0; i<pop_size/2; i++) if(inds[i] not_eq oldSrt[i]) Nswap++;
  }
  //cout << "pst:"<< print(inds) << endl;

  sampled_weights[0]->copy(weights); // first backup mean weights
  popNoiseVectors[0]->clear();       // sample 0 is always mean W, no noise
  momNois->clear(); avgNois->clear(); // prepare for
  weights->clear(); //negNois->clear(); // reductions

  #pragma omp parallel
  for(Uint i=0; i<pop_size; i++) {
    const nnReal wC = popWeights[i];
    {
      nnReal * const B = momNois->params;
      const nnReal* const Y = popNoiseVectors[ inds[i] ]->params;
      #pragma omp for simd schedule(static) aligned(B,Y : VEC_WIDTH)
      for(Uint w=0; w<pDim; w++) B[w] += wC * Y[w]*Y[w];
    }
    if(wC <=0 ) continue;
    {
      nnReal * const A = avgNois->params;
      const nnReal* const Y = popNoiseVectors[ inds[i] ]->params;
      #pragma omp for simd schedule(static) aligned(A,Y : VEC_WIDTH)
      for(Uint w=0; w<pDim; w++) A[w] += wC * Y[w];
    }
    {
      nnReal * const M = weights->params;
      const nnReal* const X = sampled_weights[ inds[i] ]->params;
      #pragma omp for simd schedule(static) aligned(M,X : VEC_WIDTH)
      for(Uint w=0; w<pDim; w++) M[w] += wC * X[w];
    }
    //{
    //  nnReal * const C = negNois->params;
    //  const nnReal* const Y = popNoiseVectors[ inds[pop_size-1-i] ]->params;
    //  #pragma omp for simd schedule(static) aligned(C,Y : VEC_WIDTH)
    //  for(Uint w=0; w<pDim; w++) C[w] += wC * Y[w];
    //}
  }

  //const nnReal * const C = negNois->params;
  const nnReal * const B = momNois->params;
  const nnReal * const A = avgNois->params;
  //nnReal * const D = pathDif->params;
  nnReal * const P = pathCov->params;
  nnReal * const S = diagCov->params;
  static constexpr nnReal c1cov = 1e-5;
  static constexpr nnReal c_sig = 1e-3;
  const nnReal alpha = 1 - c1cov - sumW*mu_eff*c1cov;
  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  #pragma omp parallel for simd schedule(static) aligned(A,B,S,P : VEC_WIDTH)
  for(Uint w=0; w<pDim; w++) {
    P[w] = (1-c_sig) * P[w] + updSigP * A[w];
    //D[w] = (1-c_sig) * D[w] + updSigP * ( A[w] - C[w] );
    S[w] = std::sqrt( alpha*S[w]*S[w] + c1cov*P[w]*P[w] + mu_eff*c1cov*B[w] );
    S[w] = std::min(S[w], (nnReal) 10); //safety
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

void CMA_Optimizer::getMetrics(ostringstream& buff)
{
  buff<<" "<<std::setw(5)<<Nswap/2; Nswap = 0; //each swap counted twice
  real2SS(buff, std::pow(diagCov->compute_weight_norm(), 2) / pDim, 6, 1);
}
void CMA_Optimizer::getHeaders(ostringstream& buff)
{
  buff << "| Nswp | avgC ";
}
