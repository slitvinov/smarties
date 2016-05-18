/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <map>

#include "../StateAction.h"
#include "NFQ.h"


NFQ::NFQ(Environment* env, Settings & settings) : Learner(env,settings), bTRAINING(settings.bTrain==1)
{ }

void NFQ::updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r)
{   // No learning here!
    //       aOld, r
    // sOld ---------> s
    //
    // Find V(s) = max Q(s, a')
    //              a'
    Real newEps;
    
    if (bTRAINING) //(false) //TODO un-hardcode
    {
        newEps = greedyEps -(greedyEps-.1)*agentId/Real(nAgents);
    }
    
    Real Vnew = Q->getMax(s, a, agentId);
    Real p = rng->uniform();
    
    double Prand = newEps;// * exp(-T->Set.size()/1e3);
    if  (p < Prand)  a.getRand(rng);
}


void NFQ::Train()
{
    const int ndata = T->Set.size();
#if 0
    if (batchSize-- <= 0 && ndata>100)
    {
        batchSize = 10000;
        Q->updateFrozenWeights();
        T->updateP();
        iter++;
        
        if (iter%100==0)
        {
            string restart_file;
            char buf[500];
            sprintf(buf, "restart.net_%09d", iter);
            restart_file = string(buf);
            Q->save(restart_file.c_str());
        }
    }
    else if(ndata>100)
    {
        int ind = T->sample();
        Real MSE = Q->Train(T->Set[ind].sOld, T->Set[ind].a, T->Set[ind].r, T->Set[ind].s, gamma, T->Ws[ind]);
        T->Errs[ind] = MSE;
    }
    
#else //the following does not prioritize "problematic" experiences
    
    if (T->inds.size()==0 && ndata>100)
    {
        T->inds.reserve(ndata);
        for (int i=0; i<ndata; ++i) T->inds.push_back(i);
        random_shuffle(T->inds.begin(), T->inds.end());
        
        Q->updateFrozenWeights();
        
        Real mean_err = accumulate(T->Errs.begin(), T->Errs.end(), 0.)/ndata;
        printf("Avg MSE %f %d\n",mean_err,ndata);
        
        if ((++T->anneal)%1000==0)
        {
            string restart_file;
            char buf[500];
            sprintf(buf, "restart.net_%09d", T->anneal);
            restart_file = string(buf);
            Q->save(restart_file.c_str());
        }
    }
    else if (T->inds.size()>0 && ndata>100) //do we have data?
    {
        const int ind = T->inds.back();
        T->inds.pop_back();
        
        Real MSE = Q->Train(T->Set[ind].sOld, T->Set[ind].a, T->Set[ind].r, T->Set[ind].s, gamma, 1.0);
        T->Errs[ind] = MSE;
    }
#endif
}
