//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"
class Aggregator;

class RETPG : public Learner_offPolicy
{
 protected:
  Aggregator* relay;
  const Uint nA = env->aI.dim;
  const Real OrUhDecay = CmaxPol<=0? .85 : 0; // as in original
  //const Real OrUhDecay = 0; // no correlated noise
  vector<Rvec> OrUhState = std::vector<Rvec>(nAgents,Rvec(nA,0));

  void TrainBySequences(const Uint seq,
    const Uint wID, const Uint bID, const Uint thrID) const override;

  void Train(const Uint seq, const Uint t,
    const Uint wID, const Uint bID, const Uint thrID) const override;

 public:
  RETPG(Environment*const _env, Settings& _set);
  ~RETPG() { }

  void select(Agent& agent) override;
};
