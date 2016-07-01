/*
 *  NFQ.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include <list>
#include "Learner.h"

using namespace std;

struct trainData
{
    trainData() : weight(1) {}
    Real weight, MSE, avgQ, minQ, maxQ, relE;
};

class NFQ : public Learner
{
    const int batchSize, tgtUpdateDelay, nAgents;
    const bool bRecurrent;
    int nInputs, nActions, nStateDims, cntUpdateDelay;
    
    vector<Real> prediction, scaledInp;
    Network* net;
    Optimizer* opt;
    Profiler* profiler;
    void TrainDQN(const vector<vector<Real>*> & sOld, const vector<int> & a, const vector<Real> & r, const vector<vector<Real>*> & s);
    void TrainRQN(const vector<int> & a, const vector<Real> & r, const vector<vector<Real>> & s);
    void updateFrozenWeights()
    {
        net->updateFrozenWeights();
        cout << profiler->printStat() << endl;
    }
public:
	NFQ(Environment* env, Settings & settings);
    bool bTRAINING,first;
    void updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r) override;
    void Train() override;
};

