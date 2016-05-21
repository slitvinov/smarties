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


NFQ::NFQ(Environment* env, Settings & settings) : Learner(env,settings), bTRAINING(settings.bTrain==1), batchSize(-1)
{ }

void NFQ::updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r)
{   // No learning here!
    //       aOld, r
    // sOld ---------> s
    //
    // Find V(s) = max Q(s, a')
    //              a'
    Real newEps(greedyEps);
    if (bTRAINING) newEps = (greedyEps-(greedyEps-.1)*agentId/Real(nAgents)) * exp(-T->Set.size()/1e3);
    
    Real Vnew = Q->getMax(s, a, agentId);
    Real p = rng->uniform();

    if  (p < newEps)  { a.getRand(rng); printf("Random action\n");}
}


void NFQ::Train()
{
    const int ndata = T->Set.size();
    if (ndata<100) return; //do we have enough data?
    
    if (batchSize <= 0)
    {
        batchSize = ndata; //
        Q->updateFrozenWeights();
    }
    batchSize--;
    
    if (T->inds.size()==0)
    {
#ifdef _Priority_
        T->updateP();
#else
        T->anneal++;
        T->inds.reserve(ndata);
        for (int i=0; i<ndata; ++i) T->inds.push_back(i);
        random_shuffle(T->inds.begin(), T->inds.end(),*(T->fix));
#endif
        
        if (T->avgQ.size()>0)
        {
            Real mean_err = accumulate(T->Errs.begin(), T->Errs.end(), 0.)/ndata;
            Real mean_Q   = accumulate(T->avgQ.begin(), T->avgQ.end(), 0.)/ndata;
            Real max_Q = *max_element(T->maxQ.begin(), T->maxQ.end());
            Real min_Q = *min_element(T->minQ.begin(), T->minQ.end());
            printf("Avg MSE %f, avg Q %f, min Q %f, max Q %f, N %d\n",
                   mean_err, mean_Q, min_Q, max_Q, ndata);
            ofstream filestats;
            filestats.open("stats.txt", ios::app);
            filestats<<T->anneal<<" "<<mean_err<<" "<<mean_Q<<" "<<max_Q<<" "<<min_Q<<endl;
            filestats.close();
        }
        
        if (T->anneal%100==0)
        {
            printf("Saving\n");
            string restart_file;
            char buf[500];
            sprintf(buf, "restart.net_%09d", T->anneal);
            restart_file = string(buf);
            Q->save(restart_file.c_str());
        }
        
        T->avgQ.resize(ndata);
        T->minQ.resize(ndata);
        T->maxQ.resize(ndata);
    }
    
#ifdef _Priority_
    const int ind = T->sample();
    Q->stats.weight=T->Ws[ind];
#else //the following does not prioritize "problematic" experiences: basic sweep
    const int ind = T->inds.back();
    Q->stats.weight=1.;
#endif
    
    T->inds.pop_back(); //dumb, dum-dumb, dum-duuuumb
    Q->Train(T->Set[ind].sOld, T->Set[ind].a, T->Set[ind].r, T->Set[ind].s);
    T->Errs[ind] = Q->stats.MSE;
    T->avgQ[ind] = Q->stats.avgQ;
    T->minQ[ind] = Q->stats.minQ;
    T->maxQ[ind] = Q->stats.maxQ;
}
