//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner_offPolicy.h"

class Discrete_policy;
class Gaussian_policy;

template<Uint nExperts>
class Gaussian_mixture;

class Discrete_advantage;
class Quadratic_advantage;

template<Uint nExperts>
class Mixture_advantage;


template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_offPolicy
{
 protected:
  // continuous actions: dimensionality of action vectors
  // discrete actions: number of options
  const Uint nA = Policy_t::compute_nA(&aInfo);
  // number of parameters of advantage approximator
  const Uint nL = Advantage_t::compute_nL(&aInfo);

  // tgtFrac_param: target fraction of off-pol samples

  // indices identifying number and starting position of the different output // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const vector<Uint> net_outputs;
  const vector<Uint> net_indices = count_indices(net_outputs);
  const vector<Uint> pol_start, adv_start;
  const Uint VsID = net_indices[0];

  // used in case of temporally correlated noise
  vector<Rvec> OrUhState = vector<Rvec>( nAgents, Rvec(nA, 0) );

  void TrainBySequences(const Uint seq, const Uint wID,
    const Uint bID, const Uint tID) const override;

  void Train(const Uint seq, const Uint samp, const Uint wID,
    const Uint bID, const Uint tID) const override;

  Rvec compute(Sequence*const traj, const Uint samp,
    const Rvec& outVec, const Policy_t& POL, const Uint thrID) const;

  Rvec offPolCorrUpdate(Sequence*const S, const Uint t,
    const Rvec output, const Policy_t& pol, const Uint thrID) const;

  inline void updateQretFront(Sequence*const S, const Uint t) const {
    if(t == 0) return;
    //called only after a new ep is added to membuf. assumed rho=1
    const Fval D = data->scaledReward(S,t) + gamma * S->state_vals[t];
    S->Q_RET[t-1] = D +gamma*(S->Q_RET[t]-S->action_adv[t]) -S->state_vals[t-1];
  }
  inline void updateQretBack(Sequence*const S, const Uint t) const {
    assert( t > 0 && not S->isLast(t) );
    const Fval W = S->offPolicImpW[t], R = data->scaledReward(S, t);
    const Fval G = gamma, C = W < 1 ? W : 1;
    const Fval D = R +G*S->state_vals[t] -S->state_vals[t-1];
    S->Q_RET[t-1] = D + G*C*(S->Q_RET[t] - S->action_adv[t]);
  }

  inline Fval updateQret(Sequence*const S, const Uint t, const Fval A,
    const Fval V, const Policy_t& pol) const {
    // shift retrace advantage with update estimate for V(s_t)
    S->setRetrace(t, S->Q_RET[t] + S->state_vals[t] -V );
    S->setStateValue(t, V); S->setAdvantage(t, A);
    //prepare Qret_{t-1} with off policy corrections for future use
    return updateQret(S, t, A, V, pol.sampImpWeight);
  }

  inline Fval updateQret(Sequence*const S, const Uint t, const Fval A,
    const Fval V, const Fval rho) const {
    assert(rho >= 0);
    if(t == 0) return 0;
    const Fval oldRet = S->Q_RET[t-1], W = rho < 1 ? rho : 1, G = gamma;
    const Fval D = data->scaledReward(S,t) +G*V - S->state_vals[t-1];
    S->setRetrace(t-1, D + G*W*(S->Q_RET[t] - A) );
    return std::fabs(S->Q_RET[t-1] - oldRet);
  }

  Rvec policyGradient(const Tuple*const _t, const Policy_t& POL,
    const Advantage_t& ADV, const Real A_RET, const Uint thrID) const;

  //inline Rvec criticGrad(const Policy_t& POL, const Advantage_t& ADV,
  //  const Real A_RET, const Real A_critic) const {
  //  const Real anneal = iter()>epsAnneal ? 1 : Real(iter())/epsAnneal;
  //  const Real varCritic = ADV.advantageVariance();
  //  const Real iEpsA = std::pow(A_RET-A_critic,2)/(varCritic+2.2e-16);
  //  const Real eta = anneal * safeExp( -0.5*iEpsA);
  //  return POL.control_grad(&ADV, eta);
  //}

  static vector<Uint> count_outputs(const ActionInfo*const aI);
  static vector<Uint> count_pol_starts(const ActionInfo*const aI);
  static vector<Uint> count_adv_starts(const ActionInfo*const aI);

 public:
  RACER(Environment*const _env, Settings& _set);
  ~RACER() { }

  void select(Agent& agent) override;

  void prepareGradient() override;
  void initializeLearner() override;
  static Uint getnOutputs(const ActionInfo*const aI);
  static Uint getnDimPolicy(const ActionInfo*const aI);
};
