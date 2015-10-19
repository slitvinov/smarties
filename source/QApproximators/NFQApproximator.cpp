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
#include "../Settings.h"
#include "../ANN/Network.h"
#include "../ANN/WaveletNet.h"

using namespace ErrorHandling;

NFQApproximator::NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, double gamma, string nettype, int nAgents) :
QApproximator(newSInfo, newActInfo), scaledInp(sInfo.dim + actInfo.dim), gamma(gamma), A(0.02), B(1.), nettype(nettype), nAgents(nAgents), actionsIt(newActInfo)
{
    rng = new RNG(rand());
    samples.Set.clear();
    
    nInputs = sInfo.dim + actInfo.dim;
    batchSize = round(settings.nnAlpha);
    
    vector<int> lsize, mblocks, mcells;
    prediction.resize(1);
    if (nettype == "ANN")
    {
        lsize.push_back(nInputs);
        lsize.push_back(13);
        lsize.push_back(13);
        lsize.push_back(7);
        lsize.push_back(1);
        ann = new Network(lsize, 0.1, 0.1, 1);
        A = .02;
        B = 1.;
    }
    else if (nettype == "WAVE")
    {
        lsize.push_back(nInputs);
        lsize.push_back(100);
        lsize.push_back(1);
        ann = new WaveletNetLM(lsize, 1);
        A = .01;
        B = 1.;
    }
    else if (nettype == "LSTM")
    {
        lsize.push_back(nInputs);
        //lsize.push_back(sInfo.dim);
        lsize.push_back(30);
        lsize.push_back(15);
        //lsize.push_back(actInfo.bounds[0]);
        lsize.push_back(1);
        //memory blocks per layer (none in input and output)
        mblocks.push_back(0);
        mblocks.push_back(15);
        mblocks.push_back(0);
        mblocks.push_back(0);
        //num mememory cell per block on layer
        mcells.push_back(0);
        mcells.push_back(1);
        mcells.push_back(0);
        mcells.push_back(0);
        
        ann = new NetworkLSTM(lsize, mblocks, mcells, 0.1, 0.2, 0.000001, 0.0, 120);
        backup.init(15,67);
        A = .01;
        B = 0.;
        //prediction.resize(actInfo.bounds[0]);
    }
}

NFQApproximator::~NFQApproximator()
{
}

double NFQApproximator::get(const State& s, const Action& a, int nAgent)
{
    s.scale(scaledInp);
    backup.copy(ann->Agents[nAgent-1]);
    if(prediction.size() < 2)
    {
        a.scale(scaledInp);
        ann->predict(scaledInp, prediction, nAgent-1);
        return prediction[0];
    }
    else
    {
        ann->Agents[nAgent-1].copy(backup);
        ann->predict(scaledInp, prediction, nAgent-1);
        return prediction[a.vals[0]];
    }
}
double NFQApproximator::advance(const State& s, const Action& a, int nAgent)
{
    s.scale(scaledInp);
    ann->Agents[nAgent-1].copy(backup);
    ann->predict(scaledInp, prediction, nAgent-1);
    
    return prediction[a.vals[0]];
}
double NFQApproximator::getMax (const State& s, int nAgent)
{
    s.scale(scaledInp);
    backup.copy(ann->Agents[nAgent-1]);
    ann->predict(scaledInp, prediction, nAgent-1);
    double Val = -1e10;
    for (int i=0; i<prediction.size(); ++i)
        Val = max(Val, prediction[i]);
    
    return Val;
}
double NFQApproximator::testMax(const State& s, int & nAct, int nAgent)
{
    s.scale(scaledInp);
    ann->predict(scaledInp, ann->Agents[nAgent-1].memory, ann->Agents[nAgent-1].ostate, ann->Agents[nAgent-1].nstate, prediction);
    
    double Val = -1e10;
    for (int i=0; i<prediction.size(); ++i)
        if (prediction[i]>Val)
        {
            nAct = i;
            Val = prediction[i];
        }
    return Val;
}
double NFQApproximator::advanceMax (const State& s, int nAgent)
{
    s.scale(scaledInp);
    ann->Agents[nAgent-1].copy(backup);
    ann->predict(scaledInp, prediction, nAgent-1);
    
    double Val = -1e10;
    for (int i=0; i<prediction.size(); ++i)
        Val = max(Val, prediction[i]);
    return Val;
}

void NFQApproximator::correct(const State& s, const Action& a, double err, int nAgent)
{
    s.scale(scaledInp);

    for (int i=0; i<prediction.size(); ++i)
        prediction[i] = 0.;
    prediction[a.vals[0]] = err;
    
    ann->improve(scaledInp, prediction, nAgent-1);
}

double NFQApproximator::test(const State& s, const Action& a, int nAgent)
{
    if (nettype == "LSTM")
    {
        s.scale(scaledInp);
        a.scale(scaledInp);
        ann->predict(scaledInp, ann->Agents[nAgent-1].memory, ann->Agents[nAgent-1].ostate, ann->Agents[nAgent-1].nstate, prediction);
        return prediction[0];
    }
    else
        return get(s, a, nAgent);
}

double NFQApproximator::batchUpdate()
{
    vector< NFQdata > pairs;
    NFQdata tmp;
    vector< double > target(1);
    Action a(actInfo);
    debug("Sample set size is %d\n", samples.Set.size());
    double err(0.0), maxo(-1e6), mino(1e6), reward;
    
    
    for (int i=0; i<samples.Set.size(); i++)
    { //target values
    
        double best = -1e10;
        actionsIt.reset();
        samples.Set[i].sNew->scale(scaledInp); //to calculate max_a (Q^{k-1} (s' , a))

        while (!actionsIt.done())
        {
            a = actionsIt.next();
            a.scale(scaledInp);
            
            ann->predict(scaledInp, prediction, 0); // scaled network curr output
            if (descale(prediction[0]) >= best + 1e-12)
            {
                best = descale(prediction[0]); // best current Q option
                actionsIt.memorize();
            }
        }
        reward = samples.Set[i].reward - fabs(samples.Set[i].sOld->vals[1]) - fabs(samples.Set[i].sOld->vals[2])/1.57079632679;
        //output i:
        target[0] = reward + gamma*best;
        tmp.outi = target; // not scaled network desired output (exact if Q has converged)
        
        //input i:
        samples.Set[i].sOld->scale(scaledInp);
        samples.Set[i].a->scale(scaledInp);
        tmp.insi = scaledInp;
        
        //old approx & scaled Q(sOld)
        ann->predict(scaledInp, prediction, 0);
        tmp.pred = prediction;
        
        maxo = max(maxo, target[0]);
        mino = min(mino, target[0]);
        
        pairs.push_back(tmp);
    }
    
    if (nettype == "ANN"){
        A = 2./(maxo - mino); //scaling factors netmax = 1, netmin = -1
        B = -A*mino -1.;
    }else if (nettype == "WAVE"){
        A = 1./(maxo - mino); //scaling factors netmax = 1, netmin = 0
        B = -A*mino -0.;
    }
    
    std::random_shuffle ( pairs.begin(), pairs.end() );
    ann->setBatchsize(pairs.size());
    
    for (int i=0; i<pairs.size(); i++)
    {
        pairs[i].outi[0] = pairs[i].pred[0] - rescale(pairs[i].outi[0]); //scaled error pred-val
        ann->improve(pairs[i].insi, pairs[i].outi, 0);
        err += fabs(pairs[i].outi[0]);
    }
    
    pairs.clear();

    debug("The average error was %f\n", err/samples.Set.size());
    return err/samples.Set.size();
}

double NFQApproximator::serialUpdate()
{
    Action a(actInfo);
    debug("Sample set size is %d\n", samples.Set.size());
    double err(0.0), reward, best, Qnew;
    vector<double> Qold(1), target(1);
    
    for (int i=0; i<samples.Set.size(); i++)
    { //target values
        //we transition to state s' and get the Qold
        backup.copy(ann->Agents[samples.Set[i].agentId-1]);
        samples.Set[i].sOld->scale(scaledInp);
        samples.Set[i].a->scale(scaledInp);
        ann->predict(scaledInp, Qold, samples.Set[i].agentId-1);
        
        samples.Set[i].sNew->scale(scaledInp); //to calculate max_a' (Q^{k-1} (s' , a'))
        actionsIt.reset();
        best = -1e10;
        while (!actionsIt.done())
        {
            a = actionsIt.next();
            a.scale(scaledInp);
            ann->predict(scaledInp, ann->Agents[samples.Set[i].agentId-1].memory, ann->Agents[samples.Set[i].agentId-1].ostate, ann->Agents[samples.Set[i].agentId-1].nstate, prediction);
            if (prediction[0] >= best + 1e-12)
            {
                best = prediction[0]; // best current Q option
                actionsIt.memorize();
            }
        }
        reward = samples.Set[i].reward - fabs(samples.Set[i].sOld->vals[1]) - fabs(samples.Set[i].sOld->vals[2])/1.57079632679;
        Qnew = reward + gamma*best;
        target[0] = 0.01*(Qnew - Qold[0]);
        err += fabs(target[0]);
        
        ann->Agents[samples.Set[i].agentId-1].copy(backup);
        samples.Set[i].sOld->scale(scaledInp);
        samples.Set[i].a->scale(scaledInp);
        ann->predict(scaledInp, Qold, samples.Set[i].agentId-1);
        
        ann->improve(scaledInp, target, samples.Set[i].agentId-1);
    }
    ann->setBatchsize(0);
    
    debug("Learning state: average error %f, sum weights %f\n", err/samples.Set.size(), ann->TotSumWeights());

    return err/samples.Set.size();
}

double NFQApproximator::serialALearning()
{
    Action a(actInfo);
    debug("Sample set size is %d\n", samples.Set.size());
    double err(0.0), maxo(-1e6), mino(1e6), Vold, Vnxt, Anew, Aold, target;
    int anxt;
    
    for (int i=0; i<samples.Set.size(); i++)
    { //target values
        //we transition to state s' and get the Qold
        //Vold = getMax(*samples.Set[i].sOld, samples.Set[i].agentId);
        Aold = get(*samples.Set[i].sOld, *samples.Set[i].a, samples.Set[i].agentId);
        Vnxt = testMax(*samples.Set[i].sNew, anxt, samples.Set[i].agentId);
        Aold = advance(*samples.Set[i].sOld, *samples.Set[i].a, samples.Set[i].agentId);
        
        double reward = samples.Set[i].reward - fabs(samples.Set[i].sOld->vals[1]) - fabs(samples.Set[i].sOld->vals[2])/1.57079632679;
        //Vold + (r + gamma*Vnew - Vold)/2. - Aold
        //Anew = Vold + (reward + gamma*Vnxt - Vold)/0.1;
        Anew = reward + gamma*Vnxt;
        target = 0.01*(Anew - Aold);
        
        maxo = max(maxo, Anew);
        mino = min(mino, Anew);
        
        correct(*samples.Set[i].sOld, *samples.Set[i].a, target, samples.Set[i].agentId);
        err += fabs(target);
    }
    ann->setBatchsize(0);
    
    A = 2./(maxo - mino); //scaling factors netmax = 1, netmin = -1
    B = -A*mino -1.;
    
    debug("Learning state: average error %f, avg weights %f, avg learn rate %f (%f - %f).\n", err/samples.Set.size(), ann->TotSumWeights(),ann->AvgLearnRate(), maxo, mino);
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

void NFQApproximator::passData(int agentId, State& sOld, Action& a, State& sNew, double reward, double altrew)
{
    debug3("+1");
    samples.add(agentId, sOld, a, sNew, reward);
}
