/*
 *  LSTMNet.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <functional>
#include "Network.h"
#include "Optimizer.h"
#include "../Profiler.h"

using namespace std;

class FishNet
{
protected:
    const int nInputs, nOutputs,nAgents;
    const bool bRecurrent;
    Profiler * profiler;
    vector<int> indexes;
    
public:
    Network * net;
    Optimizer * opt; /* ADAM optimizer */
    
    FishNet(Settings & settings);
    
    
    void predict(const vector<Real>& input, vector<Real>& output, int iAgent);
    void predict(const vector<Real>& input, vector<Real>& output);
    void predict(const vector<vector<Real>>& inputs, vector<vector<Real>>& outputs);
    void predict(const vector<Real>& S1, vector<Real>& Q1, const vector<Real>& S2, vector<Real>& Q2, int iAgent=0);
    
    void train(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, int batchsize, int nepochs);
    void train(const vector<vector<vector<Real>>>& inputs, const vector<vector<vector<Real>>>& targets, int nepochs);
    
    void resetMemories(int iAgent=0)
    {
        net->clearMemory(net->mem[iAgent]->outvals, net->mem[iAgent]->ostates);
    }
    
    void trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE);
    void trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE);
    
    void save(string fname);
    bool restart(string fname);
};