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

NFQApproximator::NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, double gamma) :
QApproximator(newSInfo, newActInfo), samples(newActInfo, newSInfo), scaledInp(sInfo.dim), actionsIt(newActInfo), gamma(gamma)
{
    rng = new RNG(rand());
    samples.Set.clear();
    // TODO: multidimensional actions
    nActions = actInfo.bounds[0];
    nStateDims = sInfo.dim;
    batchSize = round(settings.nnAlpha);
    vector<int> lsize;
    sType = WAVE;
    
    if (sType == ANN){
        lsize.push_back(nStateDims);
        lsize.push_back(14);
        lsize.push_back(14);
        lsize.push_back(7);
        lsize.push_back(1);
        for (int i=0; i<nActions; i++)
            ann.push_back(new NetworkLM(lsize, 10, batchSize));
    }else if (sType == WAVE){
        lsize.push_back(nStateDims);
        lsize.push_back(49);
        lsize.push_back(1);
        for (int i=0; i<nActions; i++)
            ann.push_back(new WaveletNetLM(lsize, 1));
    }

    prediction.resize(1);
    A.resize(nActions, 1.);
    B.resize(nActions, 0.);
}

NFQApproximator::~NFQApproximator()
{
}

double NFQApproximator::get(const State& s, const Action& a)
{
    s.scale(scaledInp);
    ann[a.vals[0]]->predict(scaledInp, prediction);
    
    return prediction[0];
}

double NFQApproximator::batchUpdate()
{
    vector< NFQdata > pairs;
    NFQdata tmp;
    vector< double > target(1);
    Action a(actInfo);
    debug("Sample set size is %d\n", samples.Set.size());
    double err(0.0), maxo(-1e6), mino(1e6);

    for (int j=0; j<actInfo.bounds[0]; j++)
    {
        a.vals[0] = j;
        for (int i=1; i<samples.Set.size(); i++)
        { //target values
            if (samples.Set[i].a->vals[0] == j)
            { //for now update one NN at the time. TODO: (0 0, 1=1, 2=-1) -> continuous!
                // output i:
                double best = -1e10;
                actionsIt.reset();
                samples.Set[i].sNew->scale(scaledInp); //to calculate max_a (Q^{k-1} (s' , a))
                
                debug4("B4 [");
                for (int i = 0; i < scaledInp.size(); ++i)
                    debug4(" %f ", scaledInp[i]);
                debug4("]\n");
                
                while (!actionsIt.done())
                {
                    ann[actionsIt.next().vals[0]]->predict(scaledInp, prediction);
                    if (descale(prediction[0],actionsIt.next().vals[0]) >= best + 1e-12)
                    {
                        best = descale(prediction[0],actionsIt.next().vals[0]);
                        actionsIt.memorize();
                    }
                }
                
                //output i:
                target[0] = samples.Set[i].reward + gamma*best;
                tmp.outi = target; // not scaled network desired output (exact if Q has converged)
                
                //input i:
                samples.Set[i].sOld->scale(scaledInp);
                tmp.insi = scaledInp;
                
                debug4("old [");
                for (int i = 0; i < scaledInp.size(); ++i)
                    debug4(" %f ", scaledInp[i]);
                debug4("]\n");
                
                //old approx & scaled Q(sOld)
                ann[j]->predict(scaledInp, prediction);
                tmp.pred = prediction;
                
                maxo = max(maxo, target[0]);
                mino = min(mino, target[0]);
                
                pairs.push_back(tmp);
            }
        }
        
        debug("Action %d (%d) has %d targets\n", a.vals[0], j, pairs.size());
        
        if (sType == ANN){
            A[j] = 2./(maxo - mino); //scaling factors netmax = 1, netmin = -1
            B[j] = -A[j]*mino -1.;
        }else if (sType == WAVE){
            A[j] = 1./(maxo - mino); //scaling factors netmax = 1, netmin = 0
            B[j] = -A[j]*mino -0.;
        }
        
        //std::random_shuffle ( pairs.begin(), pairs.end() );
        ann[j]->setBatchsize(pairs.size());
        
        for (int i=0; i<pairs.size(); i++)
        {
            pairs[i].outi[0] = pairs[i].pred[0] - rescale(pairs[i].outi[0],j); //scaled error pred-val
            ann[j]->improve(pairs[i].insi, pairs[i].outi);
            err += fabs(pairs[i].outi[0]);
        }
        pairs.clear();
        maxo=-1e6;
        mino=1e6;
    }
    debug("The error was %f\n", err/samples.Set.size());
    return err/samples.Set.size();
}

void NFQApproximator::save(string name)
{
    for (int i=0; i<nActions; i++)
    {
        string suff;
        if (sType == ANN) suff = "ANN_act";
        if (sType == WAVE) suff = "WAVE_act";
        
        ann[i]->save(name + suff + to_string(i));
    }
    
    const string morestuff = name+"restart.scaling";
    FILE * f = fopen(morestuff.c_str(), "w");
    if (f != NULL)
    {
        fprintf(f, "A[0]: %20.20e\n", A[0]);
        fprintf(f, "B[0]: %20.20e\n", B[0]);
        fprintf(f, "A[1]: %20.20e\n", A[1]);
        fprintf(f, "B[1]: %20.20e\n", B[1]);
        fprintf(f, "A[2]: %20.20e\n", A[2]);
        fprintf(f, "B[2]: %20.20e\n", B[2]);
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
    
    for (int i=0; i<nActions; i++)
    {
        string suff;
        if (sType == ANN) suff = "ANN_act";
        if (sType == WAVE) suff = "WAVE_act";
        
        res = ann[i]->restart(name + suff + to_string(i)) && res;
    }
    
    const string morestuff = name+"restart.scaling";
    FILE * f = fopen(morestuff.c_str(), "r");
    if(f != NULL)
	{
        float val;
        fscanf(f, "A[0]: %e\n", &val);
        A[0] = val;
        printf("A[0] is %e\n", A[0]);
        
        fscanf(f, "B[0]: %e\n", &val);
        B[0] = val;
        printf("B[0] is %e\n", B[0]);
        
        fscanf(f, "A[1]: %e\n", &val);
        A[1] = val;
        printf("A[1] is %e\n", A[1]);
        
        fscanf(f, "B[1]: %e\n", &val);
        B[1] = val;
        printf("B[1] is %e\n", B[1]);
        
        fscanf(f, "A[2]: %e\n", &val);
        A[2] = val;
        printf("A[2] is %e\n", A[2]);
        
        fscanf(f, "B[2]: %e\n", &val);
        B[2] = val;
        printf("B[2] is %e\n", B[2]);
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

void NFQApproximator::passData(int agentId, State& sOld, Action& a, State& sNew, double reward)
{
    debug3("+1");
    samples.add(agentId, sOld, a, sNew, reward);
}
