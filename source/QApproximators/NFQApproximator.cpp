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
#include "../ANN/LSTMNet.h"

using namespace ErrorHandling;

NFQApproximator::NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings & settings, int nAgents) :
QApproximator(newSInfo, newActInfo, settings, nAgents), scaledInp(sInfo.dim + actInfo.dim), gamma(settings.gamma), A(0.02), B(1.), nettype(settings.network), actionsIt(newActInfo), ALfac(settings.AL_fac), batchSize(0), iter(0)
{
    rng = new RNG(rand());
    
    nActions = actInfo.bounds[0];
    nStateDims = sInfo.dim;
    
    vector<int> lsize, mblocks;
    
    lsize.push_back(nStateDims);
    lsize.push_back(settings.nnLayer1);
    if (settings.nnLayer2>1 || settings.nnMemory2>1 )
    {
        lsize.push_back(settings.nnLayer2);
        if (settings.nnLayer3>1 || settings.nnMemory3>1 )
            lsize.push_back(settings.nnLayer3);
    }
    lsize.push_back(nActions);
    
    mblocks.push_back(0);
    mblocks.push_back(settings.nnMemory1);
    if (settings.nnLayer2>1 || settings.nnMemory2>1 )
    {
        mblocks.push_back(settings.nnMemory2);
        if (settings.nnLayer3>1 || settings.nnMemory3>1 )
            mblocks.push_back(settings.nnMemory3);
    }
    mblocks.push_back(0);
    
    _info("Creating the LSTM...\n");
    ann = new FishNet(lsize, mblocks, settings, nAgents);
    _info("...created.\n");
    
    A = 1-settings.gamma;
    B = 0.;
    
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
    
    s.scale(scaledInp);
    sOld.scale(scaledInpOld);
    
    ann->predict(scaledInpOld, Qold, scaledInp, Q, iAgent);
}

Real NFQApproximator::get(const State& s, const Action& a, int nAgent)
{
    s.scale(scaledInp);

    ann->predict(scaledInp, prediction, nAgent);
    return prediction[a.vals[0]];
}

Real NFQApproximator::getMax(const State& s, Action& a, int nAgent)
{
    s.scale(scaledInp);
    Real Val = -1e10;

    ann->predict(scaledInp, prediction, nAgent);
    for (int i=0; i<prediction.size(); ++i)
    {
        //printf("%f ",prediction[i]);
    if (prediction[i]>Val)
    {
        a.vals[0] = i;
        Val = prediction[i];
    }
    }
    //printf("%d\n",a.vals[0]);
    return Val;
}

void NFQApproximator::correct(const State& s, const Action& a, Real err, int nAgent)
{
    for (int i=0; i<prediction.size(); ++i)
        prediction[i] = 0.;
    prediction[a.vals[0]] = err;
    
    ann->improve(prediction, nAgent);
}

/*
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
        debug("Learning state: average error %f, avg weights %f, avg learn rate %f (%f - %f) A %f B %f.\n", err/samples.Set.size(), ann->TotSumWeights(),ann->AvgLearnRate(), maxo, mino, A, B);

        return err/samples.Set.size();
    }
*/

void NFQApproximator::Train()
{
    const int ndata = samples->Set.size();
#if 0
    if (batchSize-- <= 0 && ndata>100)
    {
        //printf("Updatingtheweights\n");
        batchSize = min(ndata,100);
        ann->updateFrozenWeights();
        samples->updateP();
        iter++;
        
        if (iter%100==0)
        {
            string restart_file;
            char buf[500];
            sprintf(buf, "restart.net_%09d", iter);
            restart_file = string(buf);
            ann->save(restart_file.c_str());
        }
    }
    else if(ndata>100)
    {
        int ind = samples->sample();
        //printf("Err prima %f ",samples->Errs[ind]);
        Real MSE = ann->trainDQ(samples->Set[ind].sOld, samples->Set[ind].a, samples->Set[ind].r, samples->Set[ind].s, gamma, samples->Ws[ind]);
        samples->Errs[ind] = MSE;
        
        //printf("Err dopo %f \n",samples->Errs[ind]);
        //printf("MSE %f iter %d sample %d weight %f\n",MSE,batchSize,ind, samples->Ws[ind]);
    }

#else

    if (indexes.size()==0 && ndata>100)
    {
        indexes.reserve(ndata);
        for (int i=0; i<ndata; ++i)
            indexes.push_back(i);
        random_shuffle(indexes.begin(), indexes.end());
        ann->updateFrozenWeights();
        //cout << ndata << endl;
        Real mean_err = accumulate(samples->Errs.begin(), samples->Errs.end(), 0.)/ndata;
        printf("Avg MSE %f %d\n",mean_err,ndata);
        //ann->trainDQ(samples->Set[indexes.back()].sOld, samples->Set[indexes.back()].a, samples->Set[indexes.back()].r, samples->Set[indexes.back()].s, gamma, 0.);
        samples->anneal++;
    }
    else if (indexes.size()>0 && ndata>100) //do we have data?
    {
        const int ind = indexes.back();
        indexes.pop_back();
        
        Real MSE = ann->trainDQ(samples->Set[ind].sOld, samples->Set[ind].a, samples->Set[ind].r, samples->Set[ind].s, gamma);
        samples->Errs[ind] = MSE;
    }
#endif
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