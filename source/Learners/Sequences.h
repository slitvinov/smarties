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
  vector<Real> s, a, mu;
  Real r = 0;
  Tuple(const Tuple*const c): s(c->s), a(c->a), mu(c->mu), r(c->r) {}
  Tuple() {}
  Tuple& operator= (const Tuple& l)
  {
    s = l.s; a = l.a; mu = l.mu; r = l.r;
    return *this;
  }
};

struct Sequence
{
  vector<Tuple*> tuples;
  int ended = 0, ID = -1, just_sampled = -1;
  Real nOffPol = 0, MSE = 0;
  vector<Real> action_adv;
  vector<Real> state_vals;
  vector<Real> Q_RET;
  //Used for sampling, filtering, and sorting off policy data:
  vector<Real> SquaredError, offPol_weight;
  vector<Real> imp_weight;

  inline Uint ndata() const {
    assert(tuples.size());
    if(tuples.size()==0) return 0;
    return tuples.size()-1;
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
  inline void setSampled(const int t)
  {
    #pragma omp critical
    if(just_sampled < t) just_sampled = t;
  }
  inline bool isOffPolicy(const Uint t,const Real W,const Real C,const Real iC)
  {
    bool isOff;
    //#pragma omp critical
    //{
      assert(t<offPol_weight.size());
      //const Real w = offPol_weight[t];
      offPol_weight[t] = W;
      //const bool wasOff = w > C || w < iC;
                  isOff = W > C || W < iC;
      //if((not wasOff)&&     isOff ) nOffPol += 1;
      //if(     wasOff &&(not isOff)) nOffPol -= 1;
    //}
    return isOff;
  }
  inline void add_state(const vector<Real> state, const Real reward=0)
  {
    Tuple * t = new Tuple();
    t->s = state; t->r = reward;
    tuples.push_back(t);
  }
  inline void add_action(const vector<Real> act,
                         const vector<Real> mu = vector<Real>())
  {
    tuples.back()->a = act;
    tuples.back()->mu = mu;
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
