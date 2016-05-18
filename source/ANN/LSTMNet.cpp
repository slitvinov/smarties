/*
 *  LSTMNet.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#include <cmath>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include "LSTMNet.h"
#include "../ErrorHandling.h"
#include <cassert>

using namespace ErrorHandling;


FishNet::FishNet(vector<int>& normalSize, vector<int>& recurrSize, Settings & settings) : nInputs(normalSize.front()), nOutputs(normalSize.back()), nAgents(settings.nAgents)
{
    profiler = new Profiler();
    net = new Network(normalSize, recurrSize, settings);
    opt = new AdamOptimizer(net, profiler, settings);
}

void FishNet::save(string fname)
{
    return net->save(fname);
}

bool FishNet::restart(string fname)
{
    return net->restart(fname);
}

void FishNet::train(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, int batchsize, int nepochs)
{
    if (inputs.size() != targets.size()) die("Mismatch between batch size of targets and inputs\n");
    
    std::chrono::time_point<std::chrono::high_resolution_clock> start,end;
    const int ndata = inputs.size();
    const int nbatches = floor((Real)ndata/batchsize);
    vector<const vector<Real>*> batch_in(batchsize), batch_out(batchsize);
    
    indexes.reserve(ndata);
    for (int i=0; i<ndata; ++i)
    {
        if (static_cast<int>(inputs[i].size()) != nInputs) die("Mismatch between size of input %d and net inputs\n",i);
        if (static_cast<int>(targets[i].size()) != nOutputs) die("Mismatch between size of output %d and net outputs\n",i);
        indexes.push_back(i);
    }
    
    for (int e=0; e<nepochs; e++)
    {
        start = std::chrono::high_resolution_clock::now();
        Real batch_err(0.), err;
        std::random_shuffle(indexes.begin(), indexes.end());
        for (int b=0; b<nbatches; ++b)
        {
            for (int i=0; i<batchsize; ++i)
            {
                batch_in[i]  =  &inputs[indexes[batchsize*b+i]];
                batch_out[i] = &targets[indexes[batchsize*b+i]];
            }
            opt->trainBatch(batch_in,batch_out,err);
            batch_err+=err;
        }
        end = std::chrono::high_resolution_clock::now();
        printf("Epoch %d/%d took %f seconds and had absolute MSE of %f. \n",e,nepochs,std::chrono::duration<Real>(end-start).count(),batch_err/ndata);
        cout << profiler->printStat() << endl;
    }
}

void FishNet::train(const vector<vector<vector<Real>>>& inputs, const vector<vector<vector<Real>>>& targets, int nepochs)
{
    if (inputs.size() != targets.size()) die("Mismatch between batch size of targets and inputs\n");
    printf("Data has size %d %d\n",inputs.size(), inputs[0].size());
    
    std::chrono::time_point<std::chrono::high_resolution_clock> start,end;
    const int ndata = inputs.size();
    vector<int> indexes;
    indexes.reserve(ndata);
    for (int i=0; i<ndata; ++i)
    {
        if (inputs[i].size() != targets[i].size()) die("Mismatch between batch size of targets and inputs\n");
        for(size_t j=0; j!=inputs[i].size(); j++)
        {
            if (static_cast<int>(inputs[i][j].size())!=nInputs) die("Mismatch between size of input %d and net inputs\n", (int)j);
            if (static_cast<int>(targets[i][j].size())!=nOutputs) die("Mismatch between size of output %d and net outputs\n", (int)j);
        }
        indexes.push_back(i);
    }
    
    for (int e=0; e<nepochs; e++)
    {
        start = std::chrono::high_resolution_clock::now();
        Real batch_err(0.), err(100.);
        std::random_shuffle(indexes.begin(), indexes.end());
        for (int b=0; b<ndata; ++b)
        {
            opt->trainSeries(inputs[indexes[b]],targets[indexes[b]],err);
            batch_err+=err;
        }
        end = std::chrono::high_resolution_clock::now();
        printf("Epoch %d/%d took %f seconds and had absolute MSE of %f. \n",e,nepochs,std::chrono::duration<Real>(end-start).count(),batch_err/ndata);
        cout << profiler->printStat() << endl;
    }
}

void FishNet::predict(const vector<Real>& input, vector<Real>& output, int iAgent)
{
    if (nInputs != static_cast<int>(   input.size())) die("Wrong input dim\n");
    if (iAgent  >= static_cast<int>(net->mem.size())) die("Wrong agent dim\n");
    
    output.resize(nOutputs); //might be a problem. Then again, I wouldn't call it MY problem
    net->expandMemory(net->mem[iAgent], net->series[0]);
    
    net->predict(input, output, net->series[0], net->series[1]);
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
}

void FishNet::predict(const vector<Real>& S1, vector<Real>& Q1, const vector<Real>& S2, vector<Real>& Q2, int iAgent)
{   //used for RL, used not to mess with mem
    if (nInputs != static_cast<int>(S1.size()) || nInputs != static_cast<int>(S2.size())) die("Wrong input dim\n");
    if (iAgent  >= static_cast<int>(net->mem.size())) die("Wrong agent dim\n");
    
    Q1.resize(nOutputs);
    Q2.resize(nOutputs);
    net->allocateSeries(2);
    net->expandMemory(net->mem[iAgent], net->series[0]);

    net->predict(S1, Q1, net->series[0], net->series[1]);
    net->predict(S2, Q2, net->series[1], net->series[2]);
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
}

void FishNet::predict(const vector<vector<Real>>& inputs, vector<vector<Real>>& outputs)
{
    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    outputs.clear();
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    for (int k=0; k<nseries; k++)
    {
        if (nInputs != static_cast<int>(inputs[k].size())) die("Wrong input %d dim\n", k);
        net->predict(inputs[k], res, net->series[0], net->series[1]);
        
        outputs.push_back(res);
        swap(net->series[0],net->series[1]);
    }
}

void FishNet::improve(const vector<Real>& error, int iAgent)
{ //bad function... should be removed really but.. legacy. ASSUMES WE FORWRD PROPPED series[1] and saved memory
    if (nOutputs != static_cast<int>(error.size())) die("Wrong errors dim\n");
    if (iAgent   >= static_cast<int>(net->mem.size())) die("Wrong agent dim\n");
    net->expandMemory(net->mem[iAgent], net->series[1]);

    net->computeGrads(error, net->series[0], net->series[1], net->grad);
    opt->update(net->grad);
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
}