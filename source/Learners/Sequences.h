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
  int ended = 0, ID = -1;
  Real MSE = 0;
  //Used by on-pol algorithms:
  vector<Real> action_adv;
  vector<Real> state_vals;
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
    tuples.clear(); ended = 0; ID = -1; MSE = 0;
    SquaredError.clear(); offPol_weight.clear();
    action_adv.clear(); state_vals.clear();
    imp_weight.clear();
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
