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
QApproximator(newSInfo, newActInfo), samples(newActInfo, newSInfo), scaledInp(sInfo.dim + actInfo.dim), actionsIt(newActInfo), gamma(gamma), A(0.02), B(1.), nettype(nettype), nAgents(nAgents)
{
    rng = new RNG(rand());
    samples.Set.clear();
    
    nInputs = sInfo.dim + actInfo.dim;
    batchSize = round(settings.nnAlpha);
    
    vector<int> lsize, mblocks, mcells;
    
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
        lsize.push_back(13);
        lsize.push_back(7);
        lsize.push_back(1);
        //memory blocks per layer (none in input and output)
        mblocks.push_back(0);
        mblocks.push_back(3);
        mblocks.push_back(3);
        mblocks.push_back(0);
        //num mememory cell per block on layer
        mcells.push_back(0);
        mcells.push_back(3);
        mcells.push_back(1);
        mcells.push_back(0);
        
        ann = new NetworkLSTM(lsize, mblocks, mcells, 0.1, 0.0, 0.001, 0.5, nAgents);
        A = .02;
        B = 1.;
    }
    lambdaold= lambdanew= errold= errnew = 0;
    first = true;
    prediction.resize(1);
}

NFQApproximator::~NFQApproximator()
{
}

double NFQApproximator::get(const State& s, const Action& a, int nAgent)
{
    s.scale(scaledInp);
    a.scale(scaledInp);
    ann->predict(scaledInp, prediction, nAgent-1);
    
    return prediction[0];
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
    double err(0.0), maxo(-1e6), mino(1e6);
    
    
    for (int i=1; i<samples.Set.size(); i++)
    { //target values
    
        double best = -1e10;
        actionsIt.reset();
        samples.Set[i].sNew->scale(scaledInp); //to calculate max_a (Q^{k-1} (s' , a))
        
        debug7("B4 [");
        for (int i = 0; i < scaledInp.size(); ++i)
            debug7(" %f ", scaledInp[i]);
        debug7("]\n");

        while (!actionsIt.done())
        {
            a = actionsIt.next();
            a.scale(scaledInp);

            debug7("%d [", a.vals[0]);
            for (int i = 0; i < scaledInp.size(); ++i)
                debug7(" %f ", scaledInp[i]);
            debug7("]\n");
            
            ann->predict(scaledInp, prediction, 0); // scaled network curr output
            if (descale(prediction[0]) >= best + 1e-12)
            {
                best = descale(prediction[0]); // best current Q option
                actionsIt.memorize();
            }
        }
        
        //output i:
        target[0] = samples.Set[i].reward + gamma*best;
        tmp.outi = target; // not scaled network desired output (exact if Q has converged)
        
        //input i:
        samples.Set[i].sOld->scale(scaledInp);
        samples.Set[i].a->scale(scaledInp);
        tmp.insi = scaledInp;
        
        debug7("old [");
        for (int i = 0; i < scaledInp.size(); ++i)
            debug7(" %f ", scaledInp[i]);
        debug7("]\n");
        
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
        pairs[i].outi[0] = rescale(pairs[i].outi[0]) - pairs[i].pred[0]; //scaled error pred-val
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
    double err(0.0);
    vector<double> Qold(1), target(1);
    
    for (int i=1; i<samples.Set.size(); i++)
    { //target values
        //we transition to state s' and get the Qold
        Memory backup(ann->Agents[samples.Set[i].agentId-1]);
        //printf("Memory %d backup example %f %f %f\n",samples.Set[i].agentId-1, backup.memory[0],  backup.ostate[0],  backup.nstate[0]);
        samples.Set[i].sOld->scale(scaledInp);
        samples.Set[i].a->scale(scaledInp);
        ann->predict(scaledInp, Qold, samples.Set[i].agentId-1);
        //_info("Qold=%f\n",Qold[0]);
        //Look for the best action by testing the NN without changin memory
        double best = -1e10;
        
        samples.Set[i].sNew->scale(scaledInp); //to calculate max_a' (Q^{k-1} (s' , a'))
        
        actionsIt.reset();
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

        target[0] = samples.Set[i].reward/100. + gamma*best - Qold[0];
        if (target[0]>1e6) {
            die("Exploding!\n");
        }
        ann->Agents[samples.Set[i].agentId-1] = backup;

        samples.Set[i].sOld->scale(scaledInp);
        samples.Set[i].a->scale(scaledInp);
        ann->predict(scaledInp, Qold, samples.Set[i].agentId-1);
        //debug("Improving on error %f\n", target[0]);
        ann->improve(scaledInp, target, samples.Set[i].agentId-1);
        err += fabs(target[0]);
    }
    ann->setBatchsize(0);
    debug("The average error was %f\n", err/samples.Set.size());
    
    if (first)
    {
        errnew = err/samples.Set.size();
        lambdanew = ann->lambda;
        ann->lambda = ann->lambda*1.01;
        delta = ann->lambda-lambdanew;
        first = false;
    }
    else
    {
        errold = errnew;
        errnew = err/samples.Set.size();
        lambdaold = lambdanew;
        lambdanew = ann->lambda;
        double change = lambdanew==lambdaold? 0.0 : -0.00001*(errnew-errold)/(lambdanew-lambdaold);
        ann->lambda = lambdanew +change +0.1*delta;
        delta = change;
        ann->lambda = ann->lambda<0? 0. : ann->lambda;
    }

    debug("We will pick lambda = %f\n", ann->lambda );
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
    
    State t_sO(sInfo), t_sN(sInfo);
    vector<double> d_sO(sInfo.dim), d_sN(sInfo.dim);
    Action t_a(actInfo);
    vector<int> d_a(actInfo.dim);
    double reward;
    int agentId;
    
    ifstream in("history.txt");
    std::string line;
    double alt_reward;
	if(in.good())
	{
		unsigned counter = 0;
		while (getline(in, line))
        {
            istringstream line_in(line);
            line_in >> agentId;
            for (int i=0; i<sInfo.dim; i++)
            {
                line_in >> d_sO[i];
            }
            for (int i=0; i<sInfo.dim; i++)
            {
                line_in >> d_sN[i];
            }
            for (int i=0; i<actInfo.dim; i++)
            {
                line_in >> d_a[i];
            }
            line_in >> reward;
            line_in >> alt_reward;
            t_sO.set(d_sO);
            t_sN.set(d_sN);
            t_a.set(d_a);
            samples.add(agentId, t_sO, t_a, t_sN, reward);
		}
	}
	else
	{
		die("WTF couldnt open file history.txt!\n");
		res = false;
	}
	
	in.close();
    return res;
}

void NFQApproximator::passData(int agentId, State& sOld, Action& a, State& sNew, double reward, double altrew)
{
    debug3("+1");
    samples.add(agentId, sOld, a, sNew, reward);
}
