/*
 * ANNApproximator.cpp
 * rl
 *
 * Created by Dmitry Alexeev on 24.06.13.
 * Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "ANNApproximator.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <cmath>

#include "../ErrorHandling.h"
#include "../Misc.h"
#include "../ANN/Network.h"
#include "../ANN/WaveletNet.h"
#include "../ANN/LSTMNet.h"

using namespace ErrorHandling;


ANNApproximator::ANNApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings settings, int nAgents) :
QApproximator(newSInfo, newActInfo), scaledInp(sInfo.dim + actInfo.dim), rng(rand()), nettype(settings.network), backup(nAgents), nAgents(nAgents)
{
    // TODO: multidimensional actions
    nActions = actInfo.bounds[0];
    nStateDims = sInfo.dim;
    nInputs = sInfo.dim + actInfo.dim;
    batchSize = round(settings.nnAlpha);
    vector<int> lsize, mblocks, mcells;
    
    if (nettype == "ANN")
    {
        (settings.nnOuts>1) ? lsize.push_back(sInfo.dim) : lsize.push_back(nInputs);
        lsize.push_back(settings.nnLayer1);
        if (settings.nnLayer2>1)
            lsize.push_back(settings.nnLayer2);
        (settings.nnOuts>1) ? lsize.push_back(actInfo.bounds[0]) : lsize.push_back(1);
        printf("Neural Network sized [%d %d %d %d]\n",lsize[0],lsize[1],lsize[2],lsize[3]);
        ann = new NetworkLM(lsize, 10, 10);
       // ann = new NetworkLM(lsize, settings.nnEta, settings.nnAlpha, settings.nnLambda, 15);
        scaledInp.resize(lsize[0]);
        prediction.resize(lsize[3]);
    }
    else if (nettype == "WAVE")
    {
        lsize.push_back(nInputs);
        lsize.push_back(settings.nnLayer1);
        lsize.push_back(1);
        printf("Wavelet Network sized [%d %d %d]\n",lsize[0],lsize[1],lsize[2]);
        ann = new WaveletNetLM(lsize, 15);

        scaledInp.resize(lsize[0]);
        prediction.resize(lsize[2]);
    }
    else if (nettype == "LSTM")
    {
        (settings.nnOuts>1) ? lsize.push_back(sInfo.dim) : lsize.push_back(nInputs);
        lsize.push_back(settings.nnLayer1);
        if (settings.nnLayer2>1)
            lsize.push_back(settings.nnLayer2);
        (settings.nnOuts>1) ? lsize.push_back(actInfo.bounds[0]) : lsize.push_back(1);
        printf("LSTM Network sized [%d %d %d %d] for %d agents\n",lsize[0],lsize[1],lsize[2],lsize[3], nAgents);
        //memory blocks per layer (none in input and output)
        mblocks.push_back(0);
        mblocks.push_back(10);
        if (settings.nnLayer2>1)
            mblocks.push_back(0);
        mblocks.push_back(0);
        //num mememory cell per block on layer

        _info("Creating the LSTM...\n");
        ann = new FishNet(lsize, mblocks, settings, nAgents);
        _info("...created.\n");
        
        scaledInp.resize(lsize[0]);
        prediction.resize(lsize.back());
    }
}

ANNApproximator::~ANNApproximator()
{
}

Real ANNApproximator::get(const State& s, const Action& a, int nAgent)
{
    s.scale(scaledInp);
    //backup.copy(ann->Agents[nAgent]);
    if(prediction.size() < 2)
    {
        a.scale(scaledInp);
        ann->predict(scaledInp, prediction, nAgent);
        return prediction[0];
    }
    else
    {
        ann->predict(scaledInp, prediction, nAgent);
        return prediction[a.vals[0]];
    }
}

Real ANNApproximator::advance(const State& s, const Action& a, int nAgent)
{
    s.scale(scaledInp);
    //ann->Agents[nAgent].copy(backup);
    if(prediction.size() < 2)
    {
        a.scale(scaledInp);
        ann->predict(scaledInp, prediction, nAgent);
        return prediction[0];
    }
    else
    {
        ann->predict(scaledInp, prediction, nAgent);
        return prediction[a.vals[0]];
    }
}

Real ANNApproximator::test(const State& s, const Action& a, int nAgent)
{
    if (nettype == "LSTM")
    {
        s.scale(scaledInp);
        if(prediction.size() < 2)
        {
            a.scale(scaledInp);
            ann->test(scaledInp, prediction, nAgent);
            return prediction[0];
        }
        else
        {
            ann->test(scaledInp, prediction, nAgent);
            return prediction[a.vals[0]];
        }
    }
    else
        return get(s, a, nAgent);
}

Real ANNApproximator::getMax (const State& s, int & nAct, int nAgent)
{
    //backup.copy(ann->Agents[nAgent]);
    s.scale(scaledInp);
    Action a(actInfo);
    Real Val = -1e10;
    if (prediction.size()>1)
    {
        ann->predict(scaledInp, prediction, nAgent);
        for (int i=0; i<prediction.size(); ++i)
            if (prediction[i]>Val)
            {
                nAct = i;
                Val = prediction[i];
            }
    }
    else
    {
        for (int i=0; i<actInfo.bounds[0]; ++i)
        {
            a.vals[0] = i;
            a.scale(scaledInp);
            ann->predict(scaledInp, prediction, nAgent);
            if (prediction[0]>Val)
            {
                nAct = i;
                Val = prediction[0];
            }
        }
    }
    return Val;
}

Real ANNApproximator::testMax (const State& s, int & nAct, int nAgent)
{
    s.scale(scaledInp);
    Real Val = -1e10;
    Action a(actInfo);
    if (prediction.size()>1)
    {
        ann->test(scaledInp, prediction, nAgent);
        for (int i=0; i<prediction.size(); ++i)
            if (prediction[i]>Val)
            {
                nAct = i;
                Val = prediction[i];
            }
    }
    else
    {
        for (int i=0; i<actInfo.bounds[0]; ++i)
        {
            a.vals[0] = i;
            a.scale(scaledInp);
            ann->test(scaledInp, prediction, nAgent);
            if (prediction[0]>Val)
            {
                nAct = i;
                Val = prediction[0];
            }
        }
    }
    return Val;
}

Real ANNApproximator::advanceMax (const State& s, int & nAct, int nAgent)
{
    //ann->Agents[nAgent].copy(backup);
    s.scale(scaledInp);
    Action a(actInfo);
    Real Val = -1e10;
    if (prediction.size()>1)
    {
        ann->predict(scaledInp, prediction, nAgent);
        for (int i=0; i<prediction.size(); ++i)
            if (prediction[i]>Val)
            {
                nAct = i;
                Val = prediction[i];
            }
    }
    else
    {
        for (int i=0; i<actInfo.bounds[0]; ++i)
        {
            a.vals[0] = i;
            a.scale(scaledInp);
            ann->predict(scaledInp, prediction, nAgent);
            if (prediction[0]>Val)
            {
                nAct = i;
                Val = prediction[0];
            }
        }
    }
    return Val;
}

void ANNApproximator::set(const State& s, const Action& a, Real val, int nAgent)
{
    s.scale(scaledInp);
    a.scale(scaledInp);
    ann->predict(scaledInp, prediction, nAgent);
    prediction[0] = nettype == "LSTM"? val - prediction[0] : prediction[0] - val;
    ann->improve(scaledInp, prediction, nAgent);
}

void ANNApproximator::correct(const State& s, const Action& a, Real err, int nAgent)
{
    s.scale(scaledInp);
    if (prediction.size()>1)
    {
        for (int i=0; i<prediction.size(); ++i)
            prediction[i] = 0.;
        prediction[a.vals[0]] = nettype == "LSTM"? err : -err;
    }
    else
    {
        a.scale(scaledInp);
        prediction[0] = nettype == "LSTM"? err : -err;
    }
    ann->improve(scaledInp, prediction, nAgent);
}

void ANNApproximator::save(string name)
{
    ann->save(name + nettype);
}

bool ANNApproximator::restart(string name)
{
    bool res = true;
    res = ann->restart(name + nettype) && res;
    return res;
}
