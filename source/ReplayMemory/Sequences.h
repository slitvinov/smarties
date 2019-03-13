//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "../StateAction.h"
#include "../Settings.h"
#include "../Environments/Environment.h"

struct Sequence
{
  Sequence()
  {
    states.reserve(MAX_SEQ_LEN);
    actions.reserve(MAX_SEQ_LEN);
    policies.reserve(MAX_SEQ_LEN);
    rewards.reserve(MAX_SEQ_LEN);
  }
  // Fval is just a storage format, probably float while Real is prob. double
  std::vector<std::vector<Fval>> states;
  std::vector<std::vector<Real>> actions;
  std::vector<std::vector<Real>> policies;
  std::vector<Real> rewards;

  // additional quantities which may be needed by algorithms:
  std::vector<Fval> Q_RET, action_adv, state_vals;
  //Used for sampling, filtering, and sorting off policy data:
  std::vector<Fval> SquaredError, offPolicImpW, KullbLeibDiv;
  std::vector<float> priorityImpW;

  // some quantities needed for processing of experiences
  long ended = 0, ID = -1, just_sampled = -1;
  Uint prefix = 0;
  Fval nOffPol = 0, MSE = 0, sumKLDiv = 0, totR = 0;

  std::mutex seq_mutex;

  inline Uint ndata() const {
    assert(states.size());
    if(states.size()==0) return 0;
    return states.size()-1;
  }
  inline bool isLast(const Uint t) const {
    return t+1 >= states.size();
  }
  inline bool isTerminal(const Uint t) const {
    return t+1 == states.size() && ended;
  }
  inline bool isTruncated(const Uint t) const {
    return t+1 == states.size() && not ended;
  }
  ~Sequence() { clear(); }
  void clear()
  {
    ended=0; ID=-1; just_sampled=-1; nOffPol=0; MSE=0; sumKLDiv=0; totR=0;
    states.clear();
    actions.clear();
    policies.clear();
    rewards.clear();
    //priorityImpW.clear();
    SquaredError.clear();
    offPolicImpW.clear();
    priorityImpW.clear();
    KullbLeibDiv.clear();
    action_adv.clear();
    state_vals.clear();
    Q_RET.clear();
  }
  inline void setSampled(const int t) {//update ind of latest sampled time step
    if(just_sampled < t) just_sampled = t;
  }
  inline void setRetrace(const Uint t, const Fval Q) {
    assert( t < Q_RET.size() );
    Q_RET[t] = Q;
  }
  inline void setAdvantage(const Uint t, const Fval A) {
    assert( t < action_adv.size() );
    action_adv[t] = A;
  }
  inline void setStateValue(const Uint t, const Fval V) {
    assert( t < state_vals.size() );
    state_vals[t] = V;
  }
  inline void setMseDklImpw(const Uint t, const Fval E, const Fval D,
    const Fval W, const Fval C, const Fval invC)
  {
    const bool wasOff = offPolicImpW[t] > C || offPolicImpW[t] < invC;
    const bool isOff = W > C || W < invC;
    {
      std::lock_guard<std::mutex> lock(seq_mutex);
      sumKLDiv = sumKLDiv - KullbLeibDiv[t] + D;
      MSE = MSE - SquaredError[t] + E;
      nOffPol = nOffPol - wasOff + isOff;
    }
    SquaredError[t] = E;
    KullbLeibDiv[t] = D;
    offPolicImpW[t] = W;
  }

  inline bool isFarPolicyPPO(const Uint t, const Fval W, const Fval C) const {
    assert(C<1) ;
    const bool isOff = W > (Fval)1 + C || W < (Fval)1 - C;
    return isOff;
  }
  inline bool isFarPolicy(const Uint t, const Fval W,
    const Fval C, const Fval invC) const {
    const bool isOff = W > C || W < invC;
    // If C<=1 assume we never filter far policy samples
    return C > (Fval)1 && isOff;
  }
  inline bool distFarPolicy(const Uint t,const Fval D,const Fval target) const {
    // If target<=0 assume we never filter far policy samples
    return target>0 && D > target;
  }
  inline void add_state(const Rvec state, const Real reward=0)
  {
    std::vector<memReal> ret( state.size(), 0 );
    for(Uint i=0; i < state.size(); i++) ret[i] = state[i];

    if(state.size()) totR += reward;
    else assert(std::fabs(reward)<2.2e-16); //rew for init state must be 0

    states.push_back(state);
    rewards.push_back(reward);
  }
  inline void add_action(const Rvec action, const Rvec mu)
  {
    actions.push_back(action);
    policies.push_back(mu);
  }
  void finalize(const Uint index)
  {
    ID = index;
    const Uint seq_len = states.size();
    // whatever the meaning of SquaredError, initialize with all zeros
    // this must be taken into account when sorting/filtering
    SquaredError = std::vector<Fval>(seq_len, 0);
    // off pol importance weights are initialized to 1s
    offPolicImpW = std::vector<Fval>(seq_len, 1);
    KullbLeibDiv = std::vector<Fval>(seq_len, 0);
  }

  int restart(FILE * f, const Uint dS, const Uint dA, const Uint dP);
  void save(FILE * f, const Uint dS, const Uint dA, const Uint dP);

  void unpackSequence(const std::vector<Fval>& data, const Uint dS,
    const Uint dA, const Uint dP);
  std::vector<Fval> packSequence(const Uint dS, const Uint dA, const Uint dP);

  static inline Uint computeTotalEpisodeSize(const Uint dS, const Uint dA,
    const Uint dP, const Uint Nstep)
  {
    const Uint tuplSize = dS+dA+dP+1;
    const Uint infoSize = 6; //adv,val,ret,mse,dkl,impW
    const Uint ret = (tuplSize+infoSize)*Nstep + 6;
    return ret;
  }
  static inline Uint computeTotalEpisodeNstep(const Uint dS, const Uint dA,
    const Uint dP, const Uint size)
  {
    const Uint tuplSize = dS+dA+dP+1;
    const Uint infoSize = 6; //adv,val,ret,mse,dkl,impW
    const Uint nStep = (size - 6)/(tuplSize+infoSize);
    assert(Sequence::computeTotalEpisodeSize(dS,dA,dP,nStep) == size);
    return nStep;
  }
};
