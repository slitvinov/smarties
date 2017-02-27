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
    const MPI_Comm mastersComm;
    const int nAgents, batchSize, tgtUpdateDelay, nThreads, nInputs, nOutputs, nAppended;
    const bool bRecurrent, bTrain;
    const Real tgtUpdateAlpha, gamma, greedyEps;
    int cntUpdateDelay, taskCounter;
    ActionInfo aInfo;
    StateInfo  sInfo;
    mt19937 * gen;
    Profiler* profiler;
    Network* net;
    Optimizer* opt;
    Transitions* data;
    trainData stats;
    vector<trainData*> Vstats;

    virtual void Train_BPTT(const int seq, const int thrID=0) const = 0;
    virtual void Train(const int seq, const int samp, const int thrID=0) const = 0;

    void dumpStats(const Real& Q, const Real& err, const vector<Real>& Qs);
    void dumpStats(trainData* const _stats, const Real& Q, const Real& err, const vector<Real>& Qs) const;
    void processStats(vector<trainData*> _stats, const Real avgTime);
    virtual void updateTargetNetwork();
    virtual void stackAndUpdateNNWeights(const int nAddedGradients);
    virtual void updateNNWeights(const int nAddedGradients);

public:
    Learner(MPI_Comm mastersComm, Environment*const env, Settings & settings);

    virtual ~Learner()
    {
        _dispose_object(profiler);
        _dispose_object(net);
        _dispose_object(opt);
        _dispose_object(data);
        for (auto & trash : Vstats) _dispose_object( trash);
    }

    virtual void select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r) = 0;
    void clearFailedSim(const int agentOne, const int agentEnd);
    bool checkBatch() const;
    void TrainBatch();
    void TrainTasking(Master* const master);
    void save(string name);
    void restart(string fname);
};
