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

using namespace ErrorHandling;

NFQApproximator::NFQApproximator(StateInfo SInfo, ActionInfo ActInfo, Settings & settings) :
QApproximator(SInfo, ActInfo, settings), iter(0), gamma(settings.gamma)
{
    nActions = actInfo.bounds[0];
    nStateDims = sInfo.dimUsed;
    
    vector<int> lsize, mblocks;
    
    lsize.push_back(nStateDims);
    lsize.push_back(settings.nnLayer1);
    if (settings.nnLayer2>1 || settings.nnMemory2>1 )
    {
        lsize.push_back(settings.nnLayer2);
        if (settings.nnLayer3>1 || settings.nnMemory3>1 )
            lsize.push_back(settings.nnLayer3);
        //lsize.push_back(0);
    }
    lsize.push_back(nActions);
    
    mblocks.push_back(0);
    mblocks.push_back(settings.nnMemory1);
    if (settings.nnLayer2>1 || settings.nnMemory2>1 )
    {
        mblocks.push_back(settings.nnMemory2);
        if (settings.nnLayer3>1 || settings.nnMemory3>1 )
            mblocks.push_back(settings.nnMemory3);
        //mblocks.push_back(settings.nnMemory3);
    }
    
    mblocks.push_back(0);
    
    profiler = new Profiler();
    net = new Network(lsize, mblocks, settings);
    opt = new AdamOptimizer(net, profiler, settings);
    
    scaledInp.resize(nStateDims);
    prediction.resize(nActions);
}

NFQApproximator::~NFQApproximator()
{
    //todo
}

void NFQApproximator::get(const State& sOld, vector<Real> & Qold, const State& s, vector<Real> & Q, int iAgent)
{
    vector<Real> scaledInpOld(nStateDims);
    
    s.scaleUsed(scaledInp);
    sOld.scaleUsed(scaledInpOld);
    Qold.resize(nActions);
    Q.resize(nActions);
    
    net->allocateSeries(2);
    net->expandMemory(net->mem[iAgent], net->series[0]);
    
    net->predict(scaledInpOld, Qold, net->series[0], net->series[1]);
    net->predict(scaledInp,    Q,    net->series[1], net->series[2]);
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
}

Real NFQApproximator::get(const State& s, const Action& a, int iAgent)
{
    s.scaleUsed(scaledInp);
    
    net->expandMemory(net->mem[iAgent], net->series[0]);
    net->predict(scaledInp, prediction, net->series[0], net->series[1]);
    net->expandMemory(net->mem[iAgent], net->series[1]);
    
    return prediction[a.vals[0]];
}

Real NFQApproximator::getMax(const State& s, Action& a, int iAgent)
{
    s.scaleUsed(scaledInp);
    Real Val = -1e10;
    
    net->expandMemory(net->mem[iAgent], net->series[0]); //used by RNN to update recurrent signals
    net->predict(scaledInp, prediction, net->series[0], net->series[1]);
#ifdef _dumpNet_
    net->dump(iAgent);
#endif
    net->expandMemory(net->mem[iAgent], net->series[1]);
    
    for (int i=0; i<prediction.size(); ++i)
    if (prediction[i]>Val)
    {
        a.vals[0] = i;
        Val = prediction[i];
    }
    
    return Val;
}

void NFQApproximator::correct(const State& s, const Action& a, Real err, int iAgent)
{
    for (int i=0; i<prediction.size(); ++i)
        prediction[i] = 0.;
    prediction[a.vals[0]] = err;
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
    net->computeGrads(prediction, net->series[0], net->series[1], net->grad);
    opt->update(net->grad);
    net->expandMemory(net->mem[iAgent], net->series[1]);
}

void NFQApproximator::Train(const vector<vector<Real>> & sOld, const vector<int> & a, const vector<Real> & r, const vector<vector<Real>> & s) //function<void(vector<Real>, st, Real, vector<Real>, vector<Real>)> & errs
{
    const int ndata = sOld.size();
    if (sOld.size()!=a.size() || sOld.size()!=r.size() || sOld.size()!=s.size()) die("Get your shit together, bro. \n");
    if(!net->allocatedFrozenWeights) die("You really should not be here\n");
    if (ndata<2) die("Series is too short \n");
    
    //cleanup memory used by the net, allocate gradient and Q outputs
    net->allocateSeries(ndata+2);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    Grads * g = new Grads(net->nWeights,net->nBiases);
    vector<Real> Qs(nActions), Qhats(nActions), Qtildes(nActions);
    stats.minQ = 1e5; stats.maxQ = -1e5;
    stats.MSE  = 0;   stats.avgQ = 0;
    if (stats.weight==0.)  opt->checkGrads(sOld);
    else
    {
        profiler->start("D");
        net->assignDropoutMask();
        profiler->stop("D");
        profiler->start("F");
        for (int k=0; k<ndata; k++) //TODO clean this shit up
        {
            if(k>0) Qs = Qhats;
            else net->predict(sOld[k], Qs, net->series[k], net->series[k+1]);
            
            if (k+1==ndata && r[k]<-0.9999) //then i reached the end-state k+1==ndata &&
            {
                for (int i=0; i<Qhats.size(); i++)
                    *(net->series[k+1]->errvals +net->iOutputs+i) = 0;
                
                Real target = r[k];
                #ifdef _scaleR_
                if (fabs(target)>1.)
                {
                    target/=fabs(target);
                    printf("Warning: big Q value %f: should be -1.\n",r[k]);
                }
                #endif
                Real err =  (target - Qs[a[k]]);
                *(net->series[k+1]->errvals +net->iOutputs +a[k]) = stats.weight*err;
                stats.MSE += err*err;
                stats.avgQ += target;
                stats.minQ = std::min(stats.minQ,target);
                stats.maxQ = std::max(stats.maxQ,target);
            }
            else
            {
                #pragma omp parallel sections
                {
                    #pragma omp section
                    net->predict(s[k], Qhats,   net->series[k+1], net->series[k+2]);
                    #pragma omp section
                    net->predict(s[k], Qtildes, net->series[k+1], net->series[ndata+2], net->frozen_weights, net->frozen_biases);
                }
                
                int Nbest; Real Vhat(-1e10);
                for (int i=0; i<Qhats.size(); i++)
                {
                    *(net->series[k+1]->errvals +net->iOutputs +i) = 0;
                    if (Qhats[i]>Vhat)  { Nbest=i; Vhat=Qhats[i]; }
                }
                
                Real target = r[k] + gamma*Qtildes[Nbest];
                #ifdef _scaleR_
                if (fabs(target)>1.)
                {
                    target/=fabs(target);
                    printf("Warning: big Q value %f (r=%f Qnext=%f)\n",r[k] + gamma*Qtildes[Nbest],r[k],Qtildes[Nbest]);
                }
                #endif
                Real err =  (target - Qs[a[k]]);
                *(net->series[k+1]->errvals +net->iOutputs +a[k]) = stats.weight*err;
                stats.MSE += err*err;
                stats.avgQ += target;
                stats.minQ = std::min(stats.minQ,target);
                stats.maxQ = std::max(stats.maxQ,target);
            }
        }
        {
            profiler->stop("F");
            //net->clearErrors(net->series[ndata+1]);
            profiler->start("B");
                net->computeDeltasEnd(net->series, ndata);
            for (int k=ndata-1; k>=1; k--)
            {
            #ifdef _BPTT_
                net->computeDeltasSeries(net->series, k);
            #else
                net->computeDeltasEnd(net->series, k);
            #endif
            }
            profiler->stop("B");
            profiler->start("G");
            for (int k=1; k<=ndata; k++)
            {
                net->computeGradsLightSeries(net->series, k, g);
                opt->stackGrads(net->grad,g);
            }
            net->removeDropoutMask(); //before the update!!!
            profiler->stop("G");
            profiler->start("O");
                opt->update(net->grad);
            profiler->stop("O");
        }
    }
    
    delete g;
    stats.MSE=std::sqrt(stats.MSE/(ndata-1));
    stats.avgQ/=ndata;
}

void NFQApproximator::save(string name)
{
    net->save(name + ".net");
}

bool NFQApproximator::restart(string name)
{
    bool res = true;
    
    res = net->restart(name + ".net") && res;

    return res;
}
