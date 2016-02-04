/*
 * NFQApproximator.cpp
 * rl
 *
 * Created by Guido Novati on 16.07.15.
 * Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "NFQApproximator.h"

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

NFQApproximator::NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings settings, int nAgents) :
QApproximator(newSInfo, newActInfo), scaledInp(sInfo.dim + actInfo.dim), gamma(settings.gamma), A(0.02), B(1.), nettype(settings.network), nAgents(nAgents), actionsIt(newActInfo), ALfac(settings.AL_fac)
{
    rng = new RNG(rand());
    samples.Set.clear();
    
    nInputs = sInfo.dim + actInfo.dim;
    batchSize = round(settings.nnAlpha);
    
    vector<int> lsize, mblocks, mcells;
    
    if (nettype == "ANN")
    {
        (settings.nnOuts>1) ? lsize.push_back(sInfo.dim) : lsize.push_back(nInputs);
        lsize.push_back(settings.nnLayer1);
        lsize.push_back(settings.nnLayer2);
        (settings.nnOuts>1) ? lsize.push_back(actInfo.bounds[0]) : lsize.push_back(1);
        printf("Neural Network sized [%d %d %d %d], ALfac=%f\n",lsize[0],lsize[1],lsize[2],lsize[3],ALfac);
        ann = new NetworkLM(lsize, 10, 120);
        A = .02;
        B = 1.;
        scaledInp.resize(lsize[0]);
        prediction.resize(lsize[3]);
    }
    else if (nettype == "WAVE")
    {
        lsize.push_back(nInputs);
        lsize.push_back(settings.nnLayer1);
        lsize.push_back(1);
        printf("Wavelet Network sized [%d %d %d], ALfac=%f\n",lsize[0],lsize[1],lsize[2],ALfac);
        ann = new WaveletNetLM(lsize, 120);
        B = 0.5;//0.5;
        A = 0.05;//.05;
        scaledInp.resize(lsize[0]);
        prediction.resize(lsize[2]);
    }
    else if (nettype == "LSTM")
    {
        (settings.nnOuts>1) ? lsize.push_back(sInfo.dim) : lsize.push_back(nInputs);
        lsize.push_back(settings.nnLayer1);
        if (settings.nnLayer2>1)
            lsize.push_back(settings.nnLayer2);
        if (settings.nnLayer3>1)
            lsize.push_back(settings.nnLayer3);
        if (settings.nnLayer4>1)
            lsize.push_back(settings.nnLayer4);
        (settings.nnOuts>1) ? lsize.push_back(actInfo.bounds[0]) : lsize.push_back(1);
        printf("LSTM Network sized [%d %d %d %d], ALfac=%f\n",lsize[0],lsize[1],lsize[2],lsize[3],ALfac);
        //memory blocks per layer (none in input and output)
        mblocks.push_back(0);
        mblocks.push_back(settings.nnMemory1);
        if (settings.nnLayer2>1)
            mblocks.push_back(settings.nnMemory2);
        if (settings.nnLayer3>1)
            mblocks.push_back(0);
        if (settings.nnLayer4>1)
            mblocks.push_back(0);
        mblocks.push_back(0);

        ann = new FishNet(lsize, mblocks, settings, 120); //TODO nagents

        A = 1-settings.gamma;
        B = 0.;
        
        scaledInp.resize(lsize[0]);
        prediction.resize(lsize.back());
    }
}

NFQApproximator::~NFQApproximator()
{
}

Real NFQApproximator::get(const State& s, const Action& a, int nAgent)
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
Real NFQApproximator::advance(const State& s, const Action& a, int nAgent)
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
Real NFQApproximator::test(const State& s, const Action& a, int nAgent)
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
Real NFQApproximator::getMax(const State& s, int & nAct, int nAgent)
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
Real NFQApproximator::testMax(const State& s, int & nAct, int nAgent)
{
    s.scale(scaledInp);
    Real Val = -1e10;
    Action a(actInfo);
    if (prediction.size()>1)
    {
        ann->test(scaledInp, prediction, nAgent);
        for (int i=0; i<prediction.size(); ++i)
        {
            //printf("Value %f for action %d\n",prediction[i],i);
            if (prediction[i]>Val)
            {
                nAct = i;
                Val = prediction[i];
            }
        }
    }
    else
    {
        for (int i=0; i<actInfo.bounds[0]; ++i)
        {
            
            a.vals[0] = i;
            a.scale(scaledInp);
            //printf("Testing action %d, %f\n",i,scaledInp.back());
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
Real NFQApproximator::advanceMax (const State& s, int & nAct, int nAgent)
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
void NFQApproximator::correct(const State& s, const Action& a, Real err, int nAgent)
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

Real NFQApproximator::batchUpdate()
{
    vector<NFQdata> pairs;
    NFQdata tmp;
    vector<Real> target(prediction.size());
    Action a(actInfo);
    debug("Sample set size is %d\n", samples.Set.size());
    
    Real err(0.0), maxo(-1e6), mino(1e6), reward, Vold, Vnxt, Aold;
    int aNxt;
    
    for (int i=0; i<samples.Set.size(); i++)
    { //target values
        //printf("aInd=%d, prediction size=%d, target size = %d\n", tmp.aInd,prediction.size(),target.size());
        if (ALfac<1.) Vold = getMax(*samples.Set[i].sOld, aNxt, samples.Set[i].agentId);
        Vnxt = testMax(*samples.Set[i].sNew, aNxt, samples.Set[i].agentId);
        
        //input i:
        samples.Set[i].sOld->scale(scaledInp);
        
        reward = 1. -samples.Set[i].sNew->vals[1]*samples.Set[i].sNew->vals[1]/.1 -samples.Set[i].sNew->vals[2]*samples.Set[i].sNew->vals[2]/.5;
        if (samples.Set[i].sNew->vals[3]<.2)
        {
            if (samples.Set[i].sNew->vals[4]==1)
                reward+=.1;
            //if (samples.Set[i].sNew->vals[4]==2)
            //    reward-=.1;
        }
        else
        {
            //if (samples.Set[i].sNew->vals[4]==1)
            //    reward-=.02;
            if (samples.Set[i].sNew->vals[4]==2)
                reward+=.1;
        }

        if (prediction.size()>1)
        {
            tmp.aInd = samples.Set[i].a->vals[0];
            for (int i=0; i<target.size(); ++i)
                target[i] = 0.;
        }
        else
        {
            tmp.aInd = 0;
            samples.Set[i].a->scale(scaledInp);
        }
        
        if (ALfac<1.) target[tmp.aInd] = descale(Vold) + (reward + gamma*descale(Vnxt) - descale(Vold))/ALfac;
        else target[tmp.aInd] = reward + gamma*descale(Vnxt);
        if (samples.Set[i].reward<-50) target[tmp.aInd] = -10;
        tmp.insi = scaledInp;
        tmp.outi = target;
        
        //maxo = max(maxo, target[tmp.aInd]);
        //mino = min(mino, target[tmp.aInd]);
        
        pairs.push_back(tmp);
    }

    std::random_shuffle ( pairs.begin(), pairs.end() ); //if only we did not have memory in LSTM...
    ann->setBatchsize(pairs.size());
    
    for (int i=0; i<pairs.size(); i++)
    {
        ann->predict(pairs[i].insi, prediction, 0); //need to put the neurons' ovals back in the right spots for the update
        pairs[i].outi[pairs[i].aInd] = prediction[pairs[i].aInd] - rescale(pairs[i].outi[pairs[i].aInd]); //scaled error pred-val

        ann->improve(pairs[i].insi, pairs[i].outi, 0);
        err += fabs(pairs[i].outi[pairs[i].aInd]);
    }
    
    pairs.clear();
    
    debug("The average error was %f, A=%f B=%f\n", err/samples.Set.size(), A,B);
    return err/samples.Set.size();
}

Real NFQApproximator::serialUpdate()
{
    Action a(actInfo);
    debug("Serial sample set size is %d\n", samples.Set.size());
    Real err(0.0), maxo(-1e6), mino(1e6), Vold, Vnxt, Anew, Aold, target, reward;
    int anxt;
    
    for (int i=0; i<samples.Set.size(); i++)
    { //target values
        //we transition to state s' and get the Qold
        //Vold = getMax(*samples.Set[i].sOld, samples.Set[i].agentId);
        Vold = getMax(*samples.Set[i].sOld, anxt, samples.Set[i].agentId);
        Vnxt = testMax(*samples.Set[i].sNew, anxt, samples.Set[i].agentId);
        Aold = advance(*samples.Set[i].sOld, *samples.Set[i].a, samples.Set[i].agentId);

        
        reward = 1. -pow(samples.Set[i].sNew->vals[1],4)/.005 -pow(samples.Set[i].sNew->vals[2],4)/.25;
        if (samples.Set[i].sNew->vals[3]<.2)
        {
            if (samples.Set[i].sNew->vals[4]==1)
                reward+=.1;
            //if (samples.Set[i].sNew->vals[4]==2)
            //    reward-=.1;
        }
        else
        {
            //if (samples.Set[i].sNew->vals[4]==1)
            //    reward-=.02;
            if (samples.Set[i].sNew->vals[4]==2)
                reward+=.1;
        }

        
        if (ALfac<1.) Anew = descale(Vold) + (reward + gamma*descale(Vnxt) - descale(Vold))/ALfac;
        else Anew = reward + gamma*descale(Vnxt);
        if (samples.Set[i].reward<-50) Anew = -10;
        //maxo = max(maxo, Anew);
        //mino = min(mino, Anew);
        
        target = (rescale(Anew) - Aold); // WTF?? check prev versions
        
        correct(*samples.Set[i].sOld, *samples.Set[i].a, target, samples.Set[i].agentId);
        err += fabs(target);
    }
    ann->setBatchsize(0);
    
    //maxo = min(maxo,  500.);
    //mino = max(mino, -500.);
    //Real goalA = 2./(maxo - mino); //scaling factors netmax = 1, netmin = -1
    //Real goalB = -A*mino -1.;
    //A += 0.1*(goalA-A); // some learn rate... 1 leads to chasing an oscillatin rescaling
    //B += 0.1*(goalB-B); // 0.01 requires too many evaluations

    debug("Learning state: average error %f, avg weights %f, avg learn rate %f (%f - %f) A %f B %f.\n", err/samples.Set.size(), ann->TotSumWeights(),ann->AvgLearnRate(), maxo, mino, A, B);

    return err/samples.Set.size();
}

void NFQApproximator::save(string name)
{

    string suff;
    ann->save(name + nettype);
    const string morestuff = name+"restart.scaling";
    FILE * f = fopen(morestuff.c_str(), "w");
    if (f != NULL)
    {
        fprintf(f, "A: %20.20e\n", A);
        fprintf(f, "B: %20.20e\n", B);
    }
    fclose(f);
    
    /*
    ofstream fout;
    fout.open("samples.txt", std::ofstream::trunc);
    debug("Sample set size is %d\n", samples.Set.size());
    for (int i=0; i<samples.Set.size(); i++)
    {
        fout << samples.Set[i].agentId << " ";
        fout << samples.Set[i].sOld->printClean().c_str();
        fout << samples.Set[i].sNew->printClean().c_str();
        fout << samples.Set[i].a->printClean().c_str();
        fout << samples.Set[i].reward << endl;
    }
    fout.close();
     */
}

bool NFQApproximator::restart(string name)
{
    bool res = true;
    
    string suff;
    res = ann->restart(name + nettype) && res;
    
    const string morestuff = name+"restart.scaling";
    FILE * f = fopen(morestuff.c_str(), "r");
    if(f != NULL)
	{
        float val;
        fscanf(f, "A: %e\n", &val);
        A = val;
        printf("A is %e\n", A);
        
        fscanf(f, "B: %e\n", &val);
        B = val;
        printf("B is %e\n", B);
        fclose(f);
    }
    
    return res;
}

void NFQApproximator::passData(int agentId, State& sOld, Action& a, State& sNew, Real reward, Real altrew)
{
    debug3("+1");
    samples.add(agentId, sOld, a, sNew, reward);
}
