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


FishNet::FishNet(vector<int>& normalSize, vector<int>& recurrSize, Settings & settings, int nAgents) : nInputs(normalSize.front()), nOutputs(normalSize.back()), nAgents(nAgents)
{
    profiler = new Profiler();

    net = new Network(normalSize, recurrSize, settings, nAgents);
    
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
        //bool far(true);
        std::random_shuffle(indexes.begin(), indexes.end());
        for (int b=0; b<ndata; ++b)
        {
            opt->checkGrads(inputs[indexes[b]],targets[indexes[b]],err);
            //if(int(err*100)%2==0)
            //    opt->trainSeries2(inputs[indexes[b]],targets[indexes[b]],err);
            //else if(int(err*100)%4==0)
            //    opt->trainSeries3(inputs[indexes[b]],targets[indexes[b]],err);
            //else
             //   opt->trainSeries(inputs[indexes[b]],targets[indexes[b]],err);
            batch_err+=err;
        }
        //if(batch_err>12) far=true; else far=false;
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
    
    #pragma omp parallel
        net->predict(input, output, net->series[0], net->series[1]);
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
}

void FishNet::predict(const vector<Real>& S1, vector<Real>& Q1, const vector<Real>& S2, vector<Real>& Q2, int iAgent)
{   //used for RL, used not to mess with mem
    if (nInputs != static_cast<int>(S1.size()) || nInputs != static_cast<int>(S2.size())) die("Wrong input dim\n");
    if (iAgent  >= static_cast<int>(net->mem.size())) die("Wrong agent dim\n");
    
    Q1.resize(nOutputs);
    Q2.resize(nOutputs);
    net->freshSeries(2);
    net->expandMemory(net->mem[iAgent], net->series[0]);
    
    #pragma omp parallel
    {
        net->predict(S1, Q1, net->series[0], net->series[1]);
        net->predict(S2, Q2, net->series[1], net->series[2]);
    }
    
    net->expandMemory(net->mem[iAgent], net->series[1]);
}

void FishNet::predict(const vector<vector<Real>>& inputs, vector<vector<Real>>& outputs)
{
    int nseries = inputs.size();
    vector<Real> res(nOutputs);
    outputs.clear();
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    #pragma omp parallel
    for (int k=0; k<nseries; k++)
    {
        if (nInputs != static_cast<int>(inputs[k].size())) die("Wrong input %d dim\n", k);
        net->predict(inputs[k], res, net->series[0], net->series[1]);
        
        #pragma omp single
        {
            outputs.push_back(res);
            swap(net->series[0],net->series[1]);
        }
    }
}

void FishNet::improve(const vector<Real>& error, int iAgent)
{ //bad function... should be removed really but.. legacy. ASSUMES WE FORWRD PROPPED series[1] and saved memory
    if (nOutputs != static_cast<int>(error.size())) die("Wrong errors dim\n");
    if (iAgent   >= static_cast<int>(net->mem.size())) die("Wrong agent dim\n");
    net->expandMemory(net->mem[iAgent], net->series[1]);
    #pragma omp parallel
    {
        net->computeGrads(error, net->series[0], net->series[1], net->grad);
        opt->update(net->grad);
    }
    net->expandMemory(net->mem[iAgent], net->series[1]);
}

/*
void FishNet::trainQ(const vector<vector<Real>> & states, const vector<int> & actions, const vector<Real> & rewards, function<void(vector<Real>, st, Real, vector<Real>)> & errs, const int iAgent)
{ //i wanna take in a function because i might decide later to use A learning instead of Q learning
    if (states.size() != actions.size() || states.size() != rewards.size()) die("Get your shit together, bro. \n");
    const int ndata = states.size();
    
    for (int k=0; k<ndata; k++)
        net->freshSeries(k+1);
    
    vector<vector<Real>> Qs(ndata, vector<Real>(nOutputs,1));
    Grads * g = new Grads(nWeights,nBiases);
    
    net->expandMemory(net->mem[iAgent], net->series[0]);
    
    #pragma omp parallel
    {
        //STEP 1: go through the data to compute predictions
        for (int k=0; k<ndata; k++)
        {
            net->predict(states[k], Qs[k], net->series[k], net->series[k+1]);
            
            #pragma omp master
            if(k>0)
            {
                errs(Qs[k-1],actions[k-1],rewards[k-1],Qs[k]); //assume lambda will put err on first arg
                for (int i=0; i<nOutputs; i++)
                *(net->series[k]->errvals +iOutputs+i) = Qs[k-1][i];
            }
        }
        
        //STEP 2: go backwards to backpropagate deltas (errors)
        net->clearErrors(net->series[ndata]); //there is a omp for in here
        for (int k=ndata-1; k>=1; k--)
            net->computeDeltasSeries(net->series, k);
        
        //STEP 3: go ahead again to compute the gradients with eligibility trace (dsdw(t) depends on dsdw(t-1))
        net->clearDsdw();
        for (int k=1; k<=ndata-1; k++)
        {
            net->computeGradsSeries(net->series, k, g);
            stackGrads(net->grad,g);
        }
        
        //STEP 4: finally update the weights
        update(net->grad);
    }
    delete g;
}
*/

#if 0
Real FishNet::trainDQ(const vector<vector<Real>> & sOld, const vector<int> & a, const vector<Real> & r, const vector<vector<Real>> & s, Real gamma, Real weight) //function<void(vector<Real>, st, Real, vector<Real>, vector<Real>)> & errs
{ //i wanna take in a function because i might decide later to use A learning instead of Q learning
    const int ndata = sOld.size();
    printf("Training with a glorious %d-length series\n",ndata);
    if (sOld.size()!=a.size() || sOld.size()!=r.size() || sOld.size()!=s.size()) die("Get your shit together, bro. \n");
    if(!net->allocatedFrozenWeights) die("You really should not be here\n");
    if (ndata<2) die("Series is too short \n");
    
    for (int k=0; k<4; ++k)
        net->freshSeries(k);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    vector<Real> Qs(nOutputs);
    vector<Real> Qhats(nOutputs);
    vector<Real> Qtildes(nOutputs);
    Real MSE = 0;
    
    #pragma omp parallel
    {
        net->clearDsdw();
        for (int k=0; k<ndata; k++)
        {
            net->predict(sOld[k], Qs, net->series[0], net->series[1]);
            //#pragma omp sections
            {
                //#pragma omp section
                    net->predict(s[k], Qhats,   net->series[1], net->series[2]);
                
                //#pragma omp section
                    net->predict(s[k], Qtildes, net->series[1], net->series[3], net->frozen_weights, net->frozen_biases);
            }
            
            #pragma omp single
            { //TODO clean this shit up
                if (k+1==ndata && r[k]<-.99)
                {
                    Real pred = Qs[a[k]];
                    for (int i=0; i<Qhats.size(); i++)
                        Qs[i] = 0;
                    Qs[a[k]] = (-1./(1.-gamma) - pred);
                    
                }
                else
                {
                    int Nbest;
                    Real Vhat(-1e10), pred(Qs[a[k]]);
                    for (int i=0; i<Qhats.size(); i++)
                    {
                        Qs[i] = 0;
                        if (Qhats[i]>Vhat)
                        {
                            Nbest = i;
                            Vhat = Qhats[i];
                        }
                    }
                    Qs[a[k]] = (r[k] + gamma*Qtildes[Nbest] -pred);
                }
                MSE += Qs[a[k]]*Qs[a[k]];
            }
            
            net->computeGrads(Qs, net->series[0], net->series[1], net->grad);
            opt->addUpdate(net->grad);
            
            #pragma omp single
                swap(net->series[0], net->series[1]);
        }
    }
    
    return MSE/ndata;
}

#else

Real FishNet::trainDQ(const vector<vector<Real>> & sOld, const vector<int> & a, const vector<Real> & r, const vector<vector<Real>> & s, Real gamma, Real weight=1.) //function<void(vector<Real>, st, Real, vector<Real>, vector<Real>)> & errs
{ //i wanna take in a function because i might decide later to use A learning instead of Q learning
    const int ndata = sOld.size();
    //printf("Training with a glorious %d-length series\n",ndata);
    if (sOld.size()!=a.size() || sOld.size()!=r.size() || sOld.size()!=s.size()) die("Get your shit together, bro. \n");
    if(!net->allocatedFrozenWeights) die("You really should not be here\n");
    if (ndata<2) die("Series is too short \n");
    
    for (int k=0; k<=ndata; k++)
        net->freshSeries(k+2);
    net->clearMemory(net->series[0]->outvals, net->series[0]->ostates);
    
    Grads * g = new Grads(net->nWeights,net->nBiases);
    
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs);
    //vector<int>  same(ndata+1);
    Real MSE = 0;

    #pragma omp parallel
    {
        for (int k=0; k<ndata; k++) //TODO clean this shit up
        {
            bool recycle(k>0);
            for (int i=0; i<nInputs && recycle; i++)
                recycle = recycle && fabs(s[k-1][i]-sOld[k][i])<1e-3;
            
            //#pragma omp master
            //same[k] = recycle; //if incoherent i must block errors
            
            //if(recycle) // recycling is good for the environment
            //{
            //    #pragma omp single
            //    Qs = Qhats;
            //}
            //else
            //{
            //    if(k>0) printf("Split series?\n");
                net->predict(sOld[k], Qs, net->series[k], net->series[k+1]);
            //}
            
            //for (int i=0; i<sOld[k].size(); i++)
            //    printf("%f ", s[k][i]);
            //printf("were the inputs (%d) (%f)\n", a[k], r[k]);

            if (k+1==ndata && r[k]<-.99) //then i reached the end-state
            {
                #pragma omp single
                {
                    for (int i=0; i<Qhats.size(); i++)
                        *(net->series[k+1]->errvals +net->iOutputs+i) = 0;
                    Real err =  (-1. - Qs[a[k]]);
                    *(net->series[k+1]->errvals +net->iOutputs +a[k]) = weight*err;
                    MSE += err*err;
                    //printf("final %f,%f,%f,%f,%f  %f \n",*(net->series[k+1]->outvals +net->iOutputs),*(net->series[k+1]->outvals +net->iOutputs+1),*(net->series[k+1]->outvals +net->iOutputs+2),*(net->series[k+1]->outvals +net->iOutputs+3),*(net->series[k+1]->outvals +net->iOutputs+4),weight*err);
                }
            }
            else
            {
                //#pragma omp sections
                {
                   // #pragma omp section
                    net->predict(s[k], Qhats,   net->series[k+1], net->series[k+2]);
                    
                   // #pragma omp section
                    net->predict(s[k], Qtildes, net->series[k+1], net->series[ndata+2], net->frozen_weights, net->frozen_biases);
                }
                
                #pragma omp single
                {
                    int Nbest; Real Vhat(-1e10);
                    for (int i=0; i<Qhats.size(); i++)
                    {
                        //printf("action %d %f %f %f \n", i, Qs[i],Qhats[i],Qtildes[i]);
                        *(net->series[k+1]->errvals +net->iOutputs +i) = 0;
                        if (Qhats[i]>Vhat)  { Nbest=i; Vhat=Qhats[i]; }
                    }
                    //printf("Best was %d \n",Nbest);
                    Real err =  (r[k]+0.5 + gamma*Qtildes[Nbest] - Qs[a[k]]);
                    *(net->series[k+1]->errvals +net->iOutputs +a[k]) = weight*err;
                    MSE += err*err;
                    //printf("%f,%f,%f,%f,%f  %f \n",*(net->series[k+1]->outvals +net->iOutputs),*(net->series[k+1]->outvals +net->iOutputs+1),*(net->series[k+1]->outvals +net->iOutputs+2),*(net->series[k+1]->outvals +net->iOutputs+3),*(net->series[k+1]->outvals +net->iOutputs+4),weight*err);
                }
            }
        }
        
        //net->clearErrors(net->series[ndata+1]); //there is a omp for
        net->computeDeltasEnd(net->series, ndata);
        for (int k=ndata-1; k>=1; k--)
            net->computeDeltasSeries(net->series, k);
        
        for (int k=1; k<=ndata; k++)
        {
            net->computeGradsLightSeries(net->series, k, g);
            opt->stackGrads(net->grad,g);
        }
        
        opt->update(net->grad);
    }
    
    delete g;
    return MSE/ndata;
}
#endif

void FishNet::updateFrozenWeights()
{
    net->updateFrozenWeights();
}