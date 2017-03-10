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
    vector<Real> a;
    Real r;

    Real SquaredError;
    Tuple(): r(0), SquaredError(0) {}
};

struct Sequence
{
    Sequence() : ended(false), MSE(0.) {}

    vector<Tuple*> tuples;
    bool ended;
    Real MSE;

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
    const MPI_Comm mastersComm;
    Environment * const env;
    const int nAppended, batchSize, maxSeqLen;
    int iOldestSaved;
    const bool bSampleSeq, bRecurrent, bWriteToFile, bNormalize, bTrain;
    const string path;
    vector<Real> std, mean;
    vector<Sequence*> Buffered;
    discrete_distribution<int> * dist;

    int add(const int agentId, const int info, const State& sOld,
             const Action& a, const State& sNew, const Real reward);

    void push_back(const int & agentId);
    void clear(const int & agentId);
    void synchronize();

public:
    int anneal, nBroken, nTransitions, nSequences, old_ndata;
    const StateInfo sI;
    const ActionInfo aI;
    Gen * gen;
    vector<Sequence*> Set, Tmp;
    vector<int> inds;

    Transitions(MPI_Comm comm, Environment*const env, Settings & settings);

    ~Transitions()
    {
        _dispose_object(gen);
        _dispose_object(dist);
        for (auto & trash : Set) _dispose_object( trash);
        for (auto & trash : Tmp) _dispose_object( trash);
        for (auto & trash : Buffered) _dispose_object( trash);
    }
    void clearFailedSim(const int agentOne, const int agentEnd);
    void pushBackEndedSim(const int agentOne, const int agentEnd);
    void update_samples_mean(const Real alpha = 0.01);
    int requestReduction(int needed) const;
    vector<Real> standardize(const vector<Real>& state, const Real noise = -1) const;
#ifdef _Priority_
    void updateP();
#endif
    void save(std::string fname);
    void restart(std::string fname);
    void updateSamples();
    int sample();
    void restartSamples();
    void saveSamples();
    int passData(const int agentId, const int info, const State & sOld,
                  const Action & a, const State & sNew, const Real reward);
};
