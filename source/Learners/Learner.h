/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "../Agent.h"
#include "../Environments/Environment.h"
#include "Transitions.h"
#include "../ANN/Network.h"
#include "../ANN/Optimizer.h"

class Master;

#include "../Scheduler.h"
#include <list>

using namespace std;

struct trainData
{
    trainData() : weight(1), MSE(0), avgQ(0), minQ(1e5), maxQ(-1e5), relE(0), dumpCount(0), epochCount(0) {}
    Real weight, MSE, avgQ, minQ, maxQ, relE;
    int dumpCount, epochCount;
};

class Learner
{
protected:
    const int nAgents, batchSize, tgtUpdateDelay, nThreads, nInputs, nOutputs;
    const bool bRecurrent, bTrain;
    const Real tgtUpdateAlpha, gamma, greedyEps;
    int cntUpdateDelay;
    ActionInfo aInfo;
    StateInfo  sInfo;
    mt19937 * gen;
    Profiler* profiler;
    Network* net;
    Optimizer* opt;
    Transitions* T;
    trainData stats;
    vector<trainData*> Vstats;
    
    virtual void Train(const int thrID, const int seq, const int first) = 0;
    virtual void Train(const int thrID, const int seq, const int samp, const int first) = 0;
    virtual void Train(const vector<int>& seq) = 0;
    virtual void Train(const vector<int>& seq, const vector<int>& samp) = 0;
    void dumpStats(const Real& Q, const Real& err, const vector<Real>& Qs);
    void dumpStats(trainData* const _stats, const Real& Q, const Real& err, const vector<Real>& Qs);
    void processStats(vector<trainData*> _stats);
public:
    vector<bool> flags;
    
    Learner(Environment* env, Settings & settings);
    virtual void select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r) = 0;
    
    bool checkBatch() const;
    void TrainBatch();
    void TrainTasking(Master* const master);
    void save(string name);
    void restart(string fname);
};
