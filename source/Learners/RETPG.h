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
  vector<Rvec> OrUhState = vector<Rvec>(nAgents,Rvec(nA,0));

  void TrainBySequences(const Uint seq,
    const Uint wID, const Uint bID, const Uint thrID) const override;

  void Train(const Uint seq, const Uint t,
    const Uint wID, const Uint bID, const Uint thrID) const override;

  inline void updateQretFront(Sequence*const S, const Uint t) const {
    if(t == 0) return;
    const Real D = data->scaledReward(S,t) + gamma*S->state_vals[t];
    S->Q_RET[t-1] = D +gamma*(S->Q_RET[t]-S->action_adv[t]) -S->state_vals[t-1];
  }
  inline void updateQretBack(Sequence*const S, const Uint t) const {
    assert( t > 0 && not S->isLast(t) );
    const Real W = S->offPolicImpW[t], R = data->scaledReward(S, t);
    const Real delta = R +gamma*S->state_vals[t] -S->state_vals[t-1];
    S->Q_RET[t-1] = delta + gamma*(W<1? W:1)*(S->Q_RET[t] - S->action_adv[t]);
  }
  inline Real updateQret(Sequence*const S, const Uint t, const Real A,
    const Real V, const Real rho) const {
    assert(rho >= 0);
    if(t == 0) return 0;
    const Real oldRet = S->Q_RET[t-1], W = rho<1 ? rho : 1;
    const Real delta = data->scaledReward(S,t) +gamma*V - S->state_vals[t-1];
    S->setRetrace(t-1, delta + gamma*W*(S->Q_RET[t] - A) );
    return std::fabs(S->Q_RET[t-1] - oldRet);
  }

 public:
  RETPG(Environment*const _env, Settings& _set);
  ~RETPG() { }

  void select(Agent& agent) override;

  void prepareGradient() override;
  void initializeLearner() override;
};
