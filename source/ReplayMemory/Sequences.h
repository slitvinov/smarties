//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Sequences_h
#define smarties_Sequences_h

#include "../Utils/Bund.h"
#include "../Utils/Warnings.h"
#include <cassert>
#include <mutex>
#include <cmath>

namespace smarties
{

inline bool isFarPolicyPPO(const Fval W, const Fval C)
{
  assert(C<1) ;
  const bool isOff = W > (Fval)1 + C || W < (Fval)1 - C;
  return isOff;
}
inline bool isFarPolicy(const Fval W, const Fval C, const Fval invC)
{
  const bool isOff = W > C || W < invC;
  // If C<=1 assume we never filter far policy samples
  return C > (Fval)1 && isOff;
}
inline bool distFarPolicy(const Fval D, const Fval target)
{
  // If target<=0 assume we never filter far policy samples
  return target>0 && D > target;
}

struct Sequence
{
  Sequence()
  {
    states.reserve(MAX_SEQ_LEN);
    actions.reserve(MAX_SEQ_LEN);
    policies.reserve(MAX_SEQ_LEN);
    rewards.reserve(MAX_SEQ_LEN);
  }

  bool isEqual(const Sequence * const S) const;

  // Fval is just a storage format, probably float while Real is prob. double
  std::vector<Fvec> states;
  std::vector<Rvec> actions;
  std::vector<Rvec> policies;
  std::vector<Real> rewards;

  // additional quantities which may be needed by algorithms:
  NNvec Q_RET, action_adv, state_vals;
  //Used for sampling, filtering, and sorting off policy data:
  Fvec SquaredError, offPolicImpW, KullbLeibDiv;
  std::vector<float> priorityImpW;

  // some quantities needed for processing of experiences
  Fval nOffPol = 0, MSE = 0, sumKLDiv = 0, totR = 0;

  // did episode terminate (i.e. terminal state) or was a time out (i.e. V(s_end) != 0
  bool ended = false;
  // unique identifier of the episode, counter
  Sint ID = -1;
  // used for prost processing eps: idx of latest time step sampled during past gradient update
  Sint just_sampled = -1;
  // used for uniform sampling : prefix sum
  Uint prefix = 0;
  // local agent id (agent id within environment) that generated epiosode
  Uint agentID;

  std::mutex seq_mutex;

  Uint ndata() const // how much data to train from? ie. not terminal
  {
    assert(states.size());
    if(states.size()==0) return 0;
    else return states.size() - 1;
  }
  Uint nsteps() const // total number of time steps observed
  {
    return states.size();
  }
  bool isTerminal(const Uint t) const
  {
    return t+1 == states.size() && ended;
  }
  bool isTruncated(const Uint t) const
  {
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
  void setSampled(const int t) //update ind of latest sampled time step
  {
    if(just_sampled < t) just_sampled = t;
  }
  void setRetrace(const Uint t, const Fval Q)
  {
    assert( t < Q_RET.size() );
    Q_RET[t] = Q;
  }
  void setAdvantage(const Uint t, const Fval A)
  {
    assert( t < action_adv.size() );
    action_adv[t] = A;
  }
  void setStateValue(const Uint t, const Fval V)
  {
    assert( t < state_vals.size() );
    state_vals[t] = V;
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

  static Uint computeTotalEpisodeSize(const Uint dS, const Uint dA,
    const Uint dP, const Uint Nstep)
  {
    const Uint tuplSize = dS+dA+dP+1;
    static constexpr Uint infoSize = 6; //adv,val,ret,mse,dkl,impW
    //extras : ended,ID,sampled,prefix,agentID x 2 for conversion safety
    static constexpr Uint extraSize = 14;
    const Uint ret = (tuplSize+infoSize)*Nstep + extraSize;
    return ret;
  }
  static Uint computeTotalEpisodeNstep(const Uint dS, const Uint dA,
    const Uint dP, const Uint size)
  {
    const Uint tuplSize = dS+dA+dP+1;
    static constexpr Uint infoSize = 6; //adv,val,ret,mse,dkl,impW
    static constexpr Uint extraSize = 14;
    const Uint nStep = (size - extraSize)/(tuplSize+infoSize);
    assert(Sequence::computeTotalEpisodeSize(dS,dA,dP,nStep) == size);
    return nStep;
  }

  void propagateRetrace(const Uint t, const Fval gamma, const Fval R)
  {
    if(t == 0) return;
    const Fval V = state_vals[t], A = action_adv[t];
    const Fval clipW = offPolicImpW[t]<1 ? offPolicImpW[t] : 1;
    Q_RET[t-1] = R + gamma * V + gamma * clipW * (Q_RET[t] - A - V);
  }
};

struct MiniBatch
{
  const Uint size;
  const Fval gamma;
  MiniBatch(const Uint _size, const Fval G) : size(_size), gamma(G)
  {
    episodes.resize(size);
    begTimeStep.resize(size);
    endTimeStep.resize(size);
    sampledTimeStep.resize(size);
    S.resize(size); R.resize(size); PERW.resize(size);
  }

  std::vector<Sequence*> episodes;
  std::vector<Uint> begTimeStep;
  std::vector<Uint> endTimeStep;
  std::vector<Uint> sampledTimeStep;
  Uint sampledBegStep(const Uint b) const { return begTimeStep[b]; }
  Uint sampledEndStep(const Uint b) const { return endTimeStep[b]; }
  Uint sampledTstep(const Uint b) const { return sampledTimeStep[b]; }
  Uint sampledNumSteps(const Uint b) const {
    assert(begTimeStep.size() > b);
    assert(endTimeStep.size() > b);
    return endTimeStep[b] - begTimeStep[b];
  }
  Uint mapTime2Ind(const Uint b, const Uint t) const
  {
    assert(begTimeStep.size() >  b);
    assert(begTimeStep[b]     <= t);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    return t - begTimeStep[b];
  }
  Uint mapInd2Time(const Uint b, const Uint k) const
  {
    assert(begTimeStep.size() > b);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    return k + begTimeStep[b];
  }

  // episodes | time steps | dimensionality
  std::vector< std::vector< NNvec > > S;  // scaled state
  std::vector< std::vector< Real  > > R;  // scaled reward
  std::vector< std::vector< nnReal> > PERW;  // prioritized sampling

  Sequence& getEpisode(const Uint b) const
  {
    return * episodes[b];
  }

  NNvec& state(const Uint b, const Uint t)
  {
    return S[b][mapTime2Ind(b, t)];
  }
  Real& reward(const Uint b, const Uint t)
  {
    return R[b][mapTime2Ind(b, t)];
  }
  nnReal& PERweight(const Uint b, const Uint t)
  {
    return PERW[b][mapTime2Ind(b, t)];
  }
  const NNvec& state(const Uint b, const Uint t) const
  {
    return S[b][mapTime2Ind(b, t)];
  }
  const Rvec& action(const Uint b, const Uint t) const
  {
    return episodes[b]->actions[t];
  }
  const Rvec& mu(const Uint b, const Uint t) const
  {
    return episodes[b]->policies[t];
  }
  const nnReal& PERweight(const Uint b, const Uint t) const
  {
    return PERW[b][mapTime2Ind(b, t)];
  }
  const Real& reward(const Uint b, const Uint t) const
  {
    return R[b][mapTime2Ind(b, t)];
  }
  nnReal& Q_RET(const Uint b, const Uint t) const
  {
    return episodes[b]->Q_RET[t];
  }
  nnReal& value(const Uint b, const Uint t) const
  {
    return episodes[b]->state_vals[t];
  }
  nnReal& advantage(const Uint b, const Uint t) const
  {
    return episodes[b]->action_adv[t];
  }


  bool isTerminal(const Uint b, const Uint t) const
  {
    return episodes[b]->isTerminal(t);
  }
  bool isTruncated(const Uint b, const Uint t) const
  {
    return episodes[b]->isTruncated(t);
  }
  Uint nTimeSteps(const Uint b) const
  {
    return episodes[b]->nsteps();
  }
  Uint nDataSteps(const Uint b) const //terminal/truncated state not actual data
  {
    return episodes[b]->ndata();
  }

  void setMseDklImpw(const Uint b, const Uint t, // batch id and time id
    const Fval E, const Fval D, const Fval W,    // error, dkl, offpol weight
    const Fval C, const Fval invC) const         // bounds of offpol weight
  {
    Sequence& EP = getEpisode(b);
    const bool wasOff = EP.offPolicImpW[t] > C || EP.offPolicImpW[t] < invC;
    const bool isOff = W > C || W < invC;
    {
      std::lock_guard<std::mutex> lock(EP.seq_mutex);
      EP.sumKLDiv = EP.sumKLDiv - EP.KullbLeibDiv[t] + D;
      EP.MSE = EP.MSE - EP.SquaredError[t] + E;
      EP.nOffPol = EP.nOffPol - wasOff + isOff;
    }
    EP.SquaredError[t] = E;
    EP.KullbLeibDiv[t] = D;
    EP.offPolicImpW[t] = W;
  }

  Fval updateRetrace(const Uint b, const Uint t,
    const Fval A, const Fval V, const Fval W) const
  {
    assert(W >= 0);
    if(t == 0) return 0; // at time 0, no reward, QRET is undefined
    Sequence& EP = getEpisode(b);
    EP.action_adv[t] = A; EP.state_vals[t] = V;
    const Fval reward = R[b][mapTime2Ind(b, t)];
    const Fval oldRet = EP.Q_RET[t-1], clipW = W<1 ? W:1;
    EP.Q_RET[t-1] = reward + gamma * V + gamma * clipW * (EP.Q_RET[t] - A - V);
    return std::fabs(EP.Q_RET[t-1] - oldRet);
  }

  void resizeStep(const Uint b, const Uint nSteps)
  {
    assert( S.size()>b); assert( R.size()>b);
    S [b].resize(nSteps); R[b].resize(nSteps); PERW[b].resize(nSteps);
  }
};
} // namespace smarties
#endif // smarties_Sequences_h
