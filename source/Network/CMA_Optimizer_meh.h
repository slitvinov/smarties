//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Optimizer.h"
#include <iomanip>
class Saru;

class CMA_Optimizer : public Optimizer
{
 protected:
  const vector<nnReal> popWeights = initializePopWeights(pop_size);
  const nnReal mu_eff = initializeMuEff(popWeights, pop_size);
  const nnReal sumW = initializeSumW(popWeights, pop_size);
  const nnReal c_sig = 1e-4; //(2 + mu_eff) / (5 + mu_eff + pDim);
  //const nnReal cpath = 0.0100; //(4 + mu_eff/pDim)/(pDim +4 +2*mu_eff/pDim);
  const nnReal c1cov = 1e-6; //2 / (mu_eff + (pDim+1.3)*(pDim+1.3) );
  nnReal anneal = std::pow( 1 - c_sig, 2 );

  const vector<Parameters*> sampled_weights;
  const vector<Parameters*> popNoiseVectors = initWpop(weights, pop_size);
  const Parameters * const avgNois = weights->allocateGrad();
  const Parameters * const momNois = weights->allocateGrad();
  const Parameters * const pathCov = weights->allocateGrad();
  const Parameters * const diagCov = weights->allocateGrad();

  vector<Saru *> generators;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  vector<Real> losses = vector<Real>(pop_size, 0);

  void initializeGeneration() const;

 public:
  nnReal sigma = 1e-2;

  CMA_Optimizer(const Settings&S, const Parameters*const W,
    const Parameters*const WT, const vector<Parameters*>&G);

  ~CMA_Optimizer();

  void prepare_update(const int BS, const vector<Rvec>& L) override;
  void apply_update() override;

  void save(const string fname, const bool bBackup) override;
  int restart(const string fname) override;

 protected:
  static inline vector<nnReal> initializePopWeights(const Uint popsz)
  {
    vector<nnReal> ret(popsz); nnReal sum = 0;
    for(Uint i=0; i<popsz; i++) {
      ret[i] = std::log(0.5*(popsz+1)) - std::log(i+1.);
      sum += std::max( ret[i], (nnReal) 0 );
    }
    for(Uint i=0; i<popsz; i++) ret[i] /= sum;
    return ret;
  }

  static inline Real initializeMuEff(const vector<nnReal>popW, const Uint popsz)
  {
    Real sum = 0, sumsq = 0;
    for(Uint i=0; i<popsz; i++) {
      const nnReal W = std::max( popW[i], (nnReal) 0 );
      sumsq += W * W; sum += W;
    }
    return sum * sum / sumsq;
  }

  static inline Real initializeSumW(const vector<nnReal>popW, const Uint popsz)
  {
    Real sum = 0;
    for(Uint i=0; i<popsz; i++) sum += popW[i];
    return sum;
  }
};
