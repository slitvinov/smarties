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
    vector<Real> s;
    int a;
    vector<Real> aC;
    Real r;
};

struct Sequence
{
    Sequence() : ended(false) {}
    
    vector<Tuple*> tuples;
    bool ended;
    
    ~Sequence()
    {
        for (auto & trash : tuples) _dispose_object( trash);
    }
};

struct Gen
{
    mt19937 * g;
    Gen(mt19937 * gen) : g(gen) { }
    size_t operator()(size_t n)
    {
        std::uniform_int_distribution<size_t> d(0, n ? n-1 : 0);
        return d(*g);
    }
};

class Transitions
{
protected:
    Environment * const env;
    const int nAppended;
    const string path;
    const bool bSampleSeq, bWriteToFile;
    vector<Real> Inp;
    vector<Sequence*> Tmp;
    discrete_distribution<int> * dist;
    
    void add(const int agentId, const int info, const State& sOld,
             const Action& a, const State& sNew, const Real reward);
    
    void push_back(const int & agentId);
    void clear(const int & agentId);
    
public:
    const StateInfo sI;
    const ActionInfo aI;
    Gen * gen;
    int anneal, nBroken, nTransitions, nSequences;
    vector<Real> Errs, Ps, Ws;
    vector<Sequence*> Set;
    vector<int> inds;
    
    Transitions(Environment* env, Settings & settings);
    
    ~Transitions()
    {
        _dispose_object(gen);
        _dispose_object(dist);
        for (auto & trash : Set) _dispose_object( trash);
        for (auto & trash : Tmp) _dispose_object( trash);
    }
    
#ifdef _Priority_
    void updateP();
#endif
    int sample();
    void restartSamples();
    void saveSamples();
    void passData(const int agentId, const int info, const State & sOld,
                  const Action & a, const State & sNew, const Real reward);
};
