/*
 *  LSTMNet.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <functional>
#include "Network.h"
#include "Optimizer.h"
#include "../Profiler.h"

using namespace std;

class FishNet: public Approximator
{
protected:
    const int nInputs, nOutputs;
    const int nAgents;
    Profiler * profiler;
    Network * net;
    Optimizer * opt; /* ADAM optimizer */
    vector<int> indexes;
    
public:
    
    FishNet(vector<int>& layerSize, vector<int>& recurSize, Settings & settings, int nAgents=1);
    
    void improve(const vector<Real>& error, int iAgent);
    
    void predict(const vector<Real>& input, vector<Real>& output, int iAgent=0);
    void predict(const vector<vector<Real>>& inputs, vector<vector<Real>>& outputs);
    void predict(const vector<Real>& S1, vector<Real>& Q1, const vector<Real>& S2, vector<Real>& Q2, int iAgent=0);
    
    void train(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, int batchsize, int nepochs);
    void train(const vector<vector<vector<Real>>>& inputs, const vector<vector<vector<Real>>>& targets, int nepochs);
    
    void resetMemories(int iAgent=0)
    {
        net->clearMemory(net->mem[iAgent]->outvals, net->mem[iAgent]->ostates);
    }
    
    Real trainDQ(const vector<vector<Real>> & sOld, const vector<int> & a, const vector<Real> & r, const vector<vector<Real>> & s, Real gamma, Real weight);
    void setBatchsize(int size) {;}
    void save(string fname);
    bool restart(string fname);
    void updateFrozenWeights();
};