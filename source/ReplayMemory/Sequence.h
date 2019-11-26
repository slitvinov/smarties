//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Sequence_h
#define smarties_Sequence_h

#include "../Utils/Bund.h"
#include "../Utils/Warnings.h"
#include <cassert>
#include <atomic>
#include <numeric>
//#include <mutex>
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
  Fval totR = 0;
  std::atomic<Uint> nFarPolicySteps{0};
  std::atomic<Fval> sumKLDivergence{0};
  std::atomic<Fval> sumSquaredErr{0};
  std::atomic<Fval> sumClipImpW{0}; // sum(min(rho,1) - 1) so we can init to 0

  void updateCumulative(const Fval C, const Fval invC)
  {
    Uint nFarPol = 0;
    Fval sumClipRho = 0;
    for (Uint t = 0; t < ndata(); ++t) {
      // float precision may cause DKL to be slightly negative:
      assert(KullbLeibDiv[t] >= - std::numeric_limits<Fval>::epsilon() && offPolicImpW[t] >= 0);
      // sequence is off policy if offPol W is out of 1/C : C
      if (offPolicImpW[t] > C || offPolicImpW[t] < invC) nFarPol += 1;
      sumClipRho += std::min((Fval) 1, offPolicImpW[t]) - 1;
    }
    nFarPolicySteps = nFarPol;
    sumClipImpW = sumClipRho;

    totR = std::accumulate(rewards.begin(), rewards.end(), 0);
    sumSquaredErr = std::accumulate(SquaredError.begin(), SquaredError.end(), 0);
    sumKLDivergence = std::accumulate(KullbLeibDiv.begin(), KullbLeibDiv.end(), 0);
  }

  void updateCumulative_atomic(const Uint t, const Fval E, const Fval D,
                               const Fval W, const Fval C, const Fval invC)
  {
    const bool wasOff = offPolicImpW[t] > C || offPolicImpW[t] < invC;
    const bool isOff  = W > C || W < invC;
    const Fval clipOldW = std::min((Fval) 1, offPolicImpW[t]);
    const Fval clipNewW = std::min((Fval) 1, W);

    sumKLDivergence.store(sumKLDivergence.load() - KullbLeibDiv[t] + D);
    sumSquaredErr.store(sumSquaredErr.load() - SquaredError[t] + E);
    sumClipImpW.store(sumClipImpW.load() - clipOldW + clipNewW);
    nFarPolicySteps += isOff - wasOff;

    SquaredError[t] = E;
    KullbLeibDiv[t] = D;
    offPolicImpW[t] = W;
  }

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
    ended = false; ID = -1; just_sampled = -1;
    nFarPolicySteps = 0;
    sumKLDivergence = 0;
    sumSquaredErr   = 0;
    sumClipImpW     = 0;
    totR = 0;

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
    static constexpr Uint infoSize = 6; //adv,val,ret, mse,dkl,impW
    //extras : ended,ID,sampled,prefix,agentID x 2 for conversion safety
    static constexpr Uint extraSize = 10;
    const Uint ret = (tuplSize+infoSize)*Nstep + extraSize;
    return ret;
  }

  static Uint computeTotalEpisodeNstep(const Uint dS, const Uint dA,
    const Uint dP, const Uint size)
  {
    const Uint tuplSize = dS+dA+dP+1;
    static constexpr Uint infoSize = 6; //adv,val,ret, mse,dkl,impW
    static constexpr Uint extraSize = 10;
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

  std::vector<float> logToFile(const Uint dimS, const Uint iterStep) const;
};

} // namespace smarties
#endif // smarties_Sequence_h
