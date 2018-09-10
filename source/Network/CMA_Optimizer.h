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
  const Real mu_eff = initializeMuEff(popWeights, pop_size);
  const nnReal c1cov = (4 + mu_eff/pDim)/(4+pDim +2*mu_eff/pDim);
  const nnReal c2cov = 2.0 / ( 1 + mu_eff + (pDim+1)*(pDim+1) );
  const nnReal cmcov = (2*mu_eff -4 +2/mu_eff)/((pDim+1)*(pDim+1) +mu_eff);
  const nnReal c_sig = (2 + mu_eff) / (5 + mu_eff + pDim);
  const nnReal updSigP = std::sqrt(c_sig * (2-c_sig) * mu_eff);
  const nnReal updPath = std::sqrt(c1cov * (2-c1cov) * mu_eff);
  const nnReal updSigm = c_sig / ( 1 + c_sig );
  const nnReal covAlph = std::sqrt(1-c2cov);
  nnReal anneal = std::pow( 1 - c_sig, 2 );

  const vector<Parameters*>& sampled_weights;
  const vector<Parameters*> popNoiseVectors = initWpop(weights, pop_size);
  const Parameters * const avgNois = weights->allocateGrad();
  const Parameters * const pathSig = weights->allocateGrad();
  const Parameters * const pathCov = weights->allocateGrad();
  const Parameters * const diagCov = weights->allocateGrad();

  vector<Saru *> generators;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  vector<Real> losses;

  void initializeGeneration() const;

 public:
  nnReal sigma = eta_init;
  
  CMA_Optimizer(Settings&S, const Parameters*const W, const Parameters*const WT,
    const vector<Parameters*>&G);

  ~CMA_Optimizer();

  void prepare_update(const int batchsize, const vector<Real> L) override;
  void apply_update() override;

  void save(const string fname, const bool bBackup) override;
  int restart(const string fname) override;
};
