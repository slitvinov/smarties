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
#include <iostream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>
#include <fstream>
#include "../Settings.h"
#include "../Environments/Environment.h"

struct Tuple
{
    State* sOld;
    Action* a;
    State* sNew;
    Real reward;
};

struct Tuples
{
    vector<vector<Real>> sOld;
    vector<vector<Real>> s;
    vector<Real> r;
    vector<int>  a;
};

struct NFQdata
{
    vector<Real> insi;
    vector<Real> outi;
    vector<Real> pred;
    int aInd;
};

class Transitions
{
protected:
    vector<Tuples> Tmp;
    vector<Real> Inp;
    StateInfo sI;
    ActionInfo aI;
    discrete_distribution<int> * dist;
    Real mean_err;
    Environment * env;
public:
    int anneal, nbroken;
    mt19937 * gen;
    vector<Real> Errs;
    vector<Tuples> Set;
    vector<Real> Ps, Ws;
    vector<int> inds;
    
    Transitions(Environment* env, Settings & settings);
    
    void add(const int & agentId, State& sOld, Action& a, State& sNew, const Real & reward);
    void push_back(const int & agentId);
    void clear(const int & agentId);
    int sample();
    void updateP();
    void restartSamples();
    void saveSamples();
    void passData(int & agentId, int & first, State & sOld, Action & a, State & sNew, Real & reward, vector<Real>& info);
};
