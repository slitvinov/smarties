//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_DPG_h
#define smarties_DPG_h

#include "Learner_approximator.h"

namespace smarties
{

struct Gaussian_policy;

class DPG : public Learner_approximator
{
  const Uint nA = env->aI.dim;
  const Real OrUhDecay = CmaxPol<=0? .85 : 0; // as in original
  //const Real OrUhDecay = 0; // no correlated noise
  std::vector<Rvec> OrUhState = std::vector<Rvec>(nAgents, Rvec(nA,0));
  Approximator* actor;
  Approximator* critc;

  void TrainBySequences(const MiniBatch& MB, const Uint wID, const Uint bID) const override;
  void Train(const MiniBatch& MB, const Uint wID, const Uint bID) const override;

public:
  DPG(Environment*const env, Settings & settings);

  void select(Agent& agent) override;
  void setupTasks(TaskQueue& tasks) override;
};

}
