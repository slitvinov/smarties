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
  typedef double memReal;
  //typedef float memReal;
  const vector<memReal> s;
  Rvec a, mu;
  const Real r;
  Tuple(const Tuple*const c): s(c->s), a(c->a), mu(c->mu), r(c->r) {}
  Tuple(const Rvec _s, const Real _r) : s(convert(_s)), r(_r) {}
  static inline vector<memReal> convert(const Rvec _s)
  {
    vector<memReal> ret ( _s.size() );
    for(Uint i=0; i < _s.size(); i++) ret[i] = _s[i];
    return ret;
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
  Rvec SquaredError, offPol_weight;
  Rvec imp_weight;

  inline Uint ndata() const {
    assert(tuples.size());
    if(tuples.size()==0) return 0;
    return tuples.size()-1;
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
    SquaredError.clear(); offPol_weight.clear();
    action_adv.clear(); state_vals.clear(); Q_RET.clear();
    imp_weight.clear();
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
    assert( t < offPol_weight.size() );
    #pragma omp atomic write
    offPol_weight[t] = W;
  }

  inline bool isOffPolicy(const Uint t,const Real W,const Real C,const Real iC)
  {
    bool isOff;
    //#pragma omp critical
    //{
      assert(t<offPol_weight.size());
      //const Real w = offPol_weight[t];
      #pragma omp atomic write
      offPol_weight[t] = W;
      //const bool wasOff = w > C || w < iC;
                  isOff = W > C || W < iC;
      //if((not wasOff)&&     isOff ) nOffPol += 1;
      //if(     wasOff &&(not isOff)) nOffPol -= 1;
    //}
    return isOff;
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
    offPol_weight = Rvec(ndata(), 1);
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
