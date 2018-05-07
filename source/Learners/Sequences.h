/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "../StateAction.h"
#include "../Settings.h"
#include "../Environments/Environment.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>

struct Tuple
{
  const vector<memReal> s;
  Rvec a, mu;
  const Real r;
  Tuple(const Tuple*const c): s(c->s), a(c->a), mu(c->mu), r(c->r) {}
  Tuple(const Rvec _s, const Real _r) : s(convert(_s)), r(_r) {}
  static inline vector<memReal> convert(const Rvec _s)
  {
    vector<memReal> ret ( _s.size() );
    for(Uint i=0; i < _s.size(); i++) ret[i] = toFinitePrec(_s[i]);
    return ret;
  }
 private:
  static inline memReal toFinitePrec(const Real val) {
    if(val > numeric_limits<float>::max()) return  numeric_limits<float>::max();
    else
    if(val <-numeric_limits<float>::max()) return -numeric_limits<float>::max();
    else
    return val;
  }
};

struct Sequence
{
  vector<Tuple*> tuples;
  int ended = 0, ID = -1, just_sampled = -1;
  Real nOffPol = 0, MSE = 0;
  Rvec action_adv;
  Rvec state_vals;
  Rvec Q_RET;
  //Used for sampling, filtering, and sorting off policy data:
  Rvec SquaredError;
  Rvec offPolicImpW;
  Rvec priorityImpW;
  //Rvec KullbLeibDiv;

  inline Uint ndata() const {
    assert(tuples.size());
    if(tuples.size()==0) return 0;
    return tuples.size()-1;
  }
  inline bool isLast(const Uint t) const {
    return t+1 >= tuples.size();
  }
  inline bool isTerminal(const Uint t) const {
    return t+1 == tuples.size() && ended;
  }
  inline bool isTruncated(const Uint t) const {
    return t+1 == tuples.size() && not ended;
  }
  ~Sequence() { clear(); }
  void clear()
  {
    for(auto &t : tuples) _dispose_object(t);
    tuples.clear();
    ended=0; ID=-1; just_sampled=-1; nOffPol=0; MSE=0;
    SquaredError.clear();
    offPolicImpW.clear();
    priorityImpW.clear();
    action_adv.clear();
    state_vals.clear();
    Q_RET.clear();
  }
  inline void setSampled(const int t) //update index of latest sampled time step
  {
    #pragma omp critical
    if(just_sampled < t) just_sampled = t;
  }
  inline void setSquaredError(const Uint t, const Real err)
  {
    assert( t < SquaredError.size() );
    #pragma omp atomic write
    SquaredError[t] = err;
  }
  inline void setRetrace(const Uint t, const Real Q)
  {
    assert( t < Q_RET.size() );
    #pragma omp atomic write
    Q_RET[t] = Q;
  }
  inline void setAdvantage(const Uint t, const Real A)
  {
    assert( t < action_adv.size() );
    #pragma omp atomic write
    action_adv[t] = A;
  }
  inline void setStateValue(const Uint t, const Real V)
  {
    assert( t < state_vals.size() );
    #pragma omp atomic write
    state_vals[t] = V;
  }
  inline void setOffPolWeight(const Uint t, const Real W)
  {
    assert( t < offPolicImpW.size() );
    #pragma omp atomic write
    offPolicImpW[t] = W;
  }

  inline bool isFarPolicyPPO(const Uint t, const Real W, const Real C)
  {
    assert(C<1) ;
    const bool isOff = W > 1+C || W < 1-C;
    assert(t<offPolicImpW.size());
    #pragma omp atomic write
    offPolicImpW[t] = W;
    return isOff;
  }
  inline bool isFarPolicy(const Uint t, const Real W, const Real C)
  {
    const bool isOff = W > C || W < 1/C;
    assert(t<offPolicImpW.size());
    #pragma omp atomic write
    offPolicImpW[t] = W;
    // If C<=1 assume we never filter far policy samples
    return C>1 && isOff;
  }
  inline bool distFarPolicy(const Uint t, const Real D, const Real W, const Real target)
  {
    assert(t<offPolicImpW.size());
    #pragma omp atomic write
    SquaredError[t] = D;
    #pragma omp atomic write
    offPolicImpW[t] = W;
    // If target<=0 assume we never filter far policy samples
    return target>0 && D > target;
  }
  inline void add_state(const Rvec state, const Real reward=0)
  {
    Tuple * t = new Tuple(state, reward);
    tuples.push_back(t);
  }
  inline void add_action(const Rvec act, const Rvec mu = Rvec())
  {
    assert( tuples.back()->s.size() && 0==tuples.back()->a.size() && 0==tuples.back()->mu.size() );
    tuples.back()->a = act;
    tuples.back()->mu = mu;
  }
  void finalize(const Uint index)
  {
    ID = index;
    // whatever the meaning of SquaredError, initialize with all zeros
    // this must be taken into account when sorting/filtering
    SquaredError = Rvec(ndata(), 0);
    // off pol importance weights are initialized to 1s
    offPolicImpW = Rvec(ndata(), 1);
  }
};

struct Gen
{
  mt19937* const g;
  Gen(mt19937 * gen) : g(gen) { }
  size_t operator()(size_t n) {
    std::uniform_int_distribution<size_t> d(0, n ? n-1 : 0);
    return d(*g);
  }
};
