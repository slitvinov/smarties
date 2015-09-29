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
#include "../Settings.h"
#include "../ANN/Network.h"
#include "../ANN/WaveletNet.h"

using namespace ErrorHandling;


ANNApproximator::ANNApproximator(StateInfo newSInfo, ActionInfo newActInfo, string tp, int nAgents) :
QApproximator(newSInfo, newActInfo), scaledInp(sInfo.dim + actInfo.dim), rng(rand()), nettype(tp), backup(nAgents), nAgents(nAgents)
{
    // TODO: multidimensional actions
    nActions = actInfo.bounds[0];
    nStateDims = sInfo.dim;
    nInputs = sInfo.dim + actInfo.dim;
    batchSize = round(settings.nnAlpha);
    vector<int> lsize, mblocks, mcells;
    
    prediction.resize(1);
    
    if (nettype == "ANN")
    {
        lsize.push_back(nInputs);
        lsize.push_back(49);
        lsize.push_back(16);
        lsize.push_back(1);
        _info("Creating the ANN...\n");
        ann = new Network(lsize, 0.5, 0.5, 0.01);
    }
    else if (nettype == "WAVE")
    {
        lsize.push_back(nInputs);
        lsize.push_back(100);
        lsize.push_back(1);
        ann = new WaveletNetLM(lsize, 1);
    }
    else if (nettype == "LSTM")
    {
        scaledInp.resize(sInfo.dim);
        prediction.resize(nActions);
        
        lsize.clear();
        lsize.push_back(nStateDims);
        lsize.push_back(20);
        lsize.push_back(10);
        lsize.push_back(nActions);
        //memory blocks per layer (none in input and output)
        mblocks.push_back(0);
        mblocks.push_back(10);
        mblocks.push_back(5);
        mblocks.push_back(0);
        //num mememory cell per block on layer
        mcells.push_back(0);
        mcells.push_back(1);
        mcells.push_back(1);
        mcells.push_back(0);
        _info("Creating the LSTM...");
        ann = new NetworkLSTM(lsize, mblocks, mcells, 0.2, 0.1, 0.01, 0.5, nAgents);
        _info("...created.");
        backup.init(15,45);
    }
}

ANNApproximator::~ANNApproximator()
{
}

double ANNApproximator::get(const State& s, const Action& a, int nAgent)
{
    if (nettype == "LSTM")
    {
        s.scale(scaledInp);
        //a.scale(scaledInp);
        backup.copy(ann->Agents[nAgent-1]);
        /*
        for (int i=0; i<backup.nMems; i++)
        {
            _info("Backup before get [%f %f %f]\n", backup.memory[i], backup.ostate[i], backup.nstate[i]);
        }
         */
        ann->predict(scaledInp, prediction, nAgent-1);
        
        return prediction[a.vals[0]];
    }
    else
    {
        s.scale(scaledInp);
        a.scale(scaledInp);
        ann->predict(scaledInp, prediction, nAgent-1);
        
        return prediction[0];
    }
}

double ANNApproximator::advance(const State& s, const Action& a, int nAgent)
{
    if (nettype == "LSTM")
    {
        s.scale(scaledInp);
        //a.scale(scaledInp);
        /*
        for (int i=0; i<backup.nMems; i++)
        {
            _info("Backup before advance %d [%f %f %f]\n",nAgent, backup.memory[i], backup.ostate[i], backup.nstate[i]);
            _info("Backup before getMax %d [%f %f %f]\n", nAgent, ann->Agents[nAgent-1].memory[i], ann->Agents[nAgent-1].ostate[i], ann->Agents[nAgent-1].nstate[i]);
        }
         */
        ann->Agents[nAgent-1].copy(backup);
        
        ann->predict(scaledInp, prediction, nAgent-1);
        return prediction[a.vals[0]];
    }
    else
        return get(s, a, nAgent);
}

double ANNApproximator::test(const State& s, const Action& a, int nAgent)
{
    if (nettype == "LSTM")
    {
        s.scale(scaledInp);
        //a.scale(scaledInp);
        ann->predict(scaledInp, ann->Agents[nAgent-1].memory, ann->Agents[nAgent-1].ostate, ann->Agents[nAgent-1].nstate, prediction);
        return prediction[a.vals[0]];
    }
    else
       return get(s, a, nAgent);
}

double ANNApproximator::getMax (const State& s, int nAgent)
{
    s.scale(scaledInp);
    backup.copy(ann->Agents[nAgent-1]);
    /*
    for (int i=0; i<backup.nMems; i++)
    {
        //_info("Backup before getMax %d [%f %f %f]\n", nAgent, backup.memory[i], backup.ostate[i], backup.nstate[i]);
        
    }
     */
    ann->predict(scaledInp, prediction, nAgent-1);
    double Val = -1e10;
    for (int i=0; i<prediction.size(); ++i)
        Val = max(Val, prediction[i]);
    
    return Val;
}

double ANNApproximator::testMax (const State& s, int & nAct, int nAgent)
{
    s.scale(scaledInp);
    /*
    for (int i=0; i<backup.nMems; i++)
    {
        //_info("Agents before testMax %d [%f %f %f]\n", nAgent, ann->Agents[nAgent-1].memory[i], ann->Agents[nAgent-1].ostate[i], ann->Agents[nAgent-1].nstate[i]);
    }
    */
    ann->predict(scaledInp, ann->Agents[nAgent-1].memory, ann->Agents[nAgent-1].ostate, ann->Agents[nAgent-1].nstate, prediction);
    
    double Val = -1e10;
    for (int i=0; i<prediction.size(); ++i)
        if (prediction[i]>Val)
        {
            nAct = i;
            Val = prediction[i];
            //printf("%f\n", Val);
        }
    //printf("(ANN) Chosen action %d\n",nAct);
    //printf("\n");
    return Val;
}

double ANNApproximator::advanceMax (const State& s, int nAgent)
{
    s.scale(scaledInp);
    /*
    for (int i=0; i<backup.nMems; i++)
    {
        //_info("Backup before advanceMax %d [%f %f %f]\n", nAgent, backup.memory[i], backup.ostate[i], backup.nstate[i]);
    }
     */
    ann->Agents[nAgent-1].copy(backup);
    ann->predict(scaledInp, prediction, nAgent-1);
    
    double Val = -1e10;
    for (int i=0; i<prediction.size(); ++i)
        Val = max(Val, prediction[i]);
    return Val;
}

void ANNApproximator::set(const State& s, const Action& a, double val, int nAgent)
{
    s.scale(scaledInp);
    a.scale(scaledInp);
    ann->predict(scaledInp, prediction, nAgent-1);
    prediction[0] = nettype == "LSTM"? val - prediction[0] : prediction[0] - val;
    ann->improve(scaledInp, prediction, nAgent-1);
}

void ANNApproximator::correct(const State& s, const Action& a, double err, int nAgent)
{
    s.scale(scaledInp);
    
    if (nettype == "LSTM")
    {
        for (int i=0; i<prediction.size(); ++i)
            prediction[i] = 0.;
        prediction[a.vals[0]] = err;
    }
    else
    {
        a.scale(scaledInp);
        prediction[0] = -err;
    }
    
    ann->improve(scaledInp, prediction, nAgent-1);
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
