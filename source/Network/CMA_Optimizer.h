//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Parameters.h"
#include <iomanip>


class CMA_Optimizer
{
 protected:
  const MPI_Comm mastersComm;
  const Uint learn_size, pop_size;
  const Parameters * const weights;
  const Parameters * const tgt_weights;
  const Uint pDim = weights->nParams;
  const vector<nnReal> popWeights = initializePopWeights(pop_size);
  const Real sigma_init, mu_eff = initializeMuEff(popWeights);

  const nnReal c1cov = (4 + mu_eff/pDim)/(4+pDim +2*mu_eff/pDim);
  const nnReal c2cov = 2.0 / ( 1 + mu_eff + (pDim+1)*(pDim+1) );
  const nnReal cmcov = (2*mu_eff -4 +2/mu_eff)/((pDim+1)*(pDim+1) +mu_eff);
  const nnReal c_sig = (2 + mu_eff) / (5 + mu_eff + pDim);
  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  const nnReal updPath = std::sqrt(c1cov * (2-c1cov) * mu_eff);
  const nnReal updSigm = c_sig / ( 1 + c_sig );
  const nnReal covAlph = std::sqrt(1-c2cov);

  Real anneal = std::pow( 1 - c_sig, 2 );

  vector<Parameters*> sampled_weights = initWpop(weights, pop_size);
  vector<Parameters*> popNoiseVectors = initWpop(weights, pop_size);

  const Parameters * const avgNois = weights->allocateGrad();
  const Parameters * const pathSig = weights->allocateGrad();
  const Parameters * const pathCov = weights->allocateGrad();
  const Parameters * const diagCov = weights->allocateGrad();

  vector<Saru *> generators;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  vector<Real> losses;

  static vector<nnReal> initializePopWeights(const Uint popsz) {
    vector<nnReal> ret(popsz); nnReal sum = 0;
    for(Uint i=0; i<popsz; i++) {
      ret[i] = std::max( std::log(0.5*(popsz+1)) - std::log(i+1.), 0.0 );
      sum += ret[i];
    }
    for(Uint i=0; i<popsz; i++) ret[i] /= sum;
    return ret;
  }

  static Real initializeMuEff(const vector<nnReal> popW) {
    Real sum = 0, sumsq = 0;
    for(Uint i=0; i<popsz; i++) {
      sumsq += popW[i] * popW[i];
      sum += popW[i];
    }
    return sum * sum / sumsq;
  }

  static vector<Parameters*> initWpop(const Parameters*const W, Uint popsz) {
    vector<Parameters*> ret(popsz, nullptr);
    for(Uint i=0; i<popsz; i++) ret[i] = W->allocateGrad();
    return ret;
  }

 public:
  bool bAnnealLearnRate = true;
  long unsigned nStep = 0;
  Real sigma = sigma_init;

  CMA_Optimizer(Settings&S, const Parameters*const W,
    const Parameters*const W_TGT);

  virtual ~CMA_Optimizer();

  void initializeGeneration() const;

  void prepare_update(const vector<Real> losses);

  void apply_update();

  void save(const string fname, const bool bBackup);
  int restart(const string fname);
};
