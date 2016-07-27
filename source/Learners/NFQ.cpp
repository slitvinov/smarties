/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "NFQ.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cmath>

NFQ::NFQ(Environment* env, Settings & settings) : Learner(env,settings)
{
}

void NFQ::select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r)
{
    vector<Real> output(nOutputs), inputs(nInputs);
    s.scaleUsed(inputs);
    if (info==1) //is it the first action performed by agent?
    {   //then old state, old action and reward are meaningless, just give the action
        net->predict(inputs, output, net->series[1]);
    }
    else
    {   //then if i'm using RNN i need to load recurrent connections
        net->expandMemory(net->mem[agentId], net->series[0]);
        net->predict(inputs, output, net->series[0], net->series[1]);
        //also, store sOld, aOld -> sNew, r
        
        data->passData(agentId, info, sOld, aOld, s, r);
    }
    #ifdef _dumpNet_
    net->dump(agentId);
    #endif
    
    //save network transition
    net->expandMemory(net->mem[agentId], net->series[1]);
    
    //load computed policy into a
    Real Val(-1e6); int Nbest;
    for (int i=0; i<nOutputs; ++i) {
        if (output[i]>Val) { Nbest=i; Val=output[i]; }
    }
    a.unpack(Nbest);
    
    //printf("Net selected %d %f for state %s\n",Nbest,a.valsContinuous[0],s.print().c_str());
    //random action?
    Real newEps(greedyEps);
    if (bTrain) { //if training: anneal random chance if i'm just starting to learn
        const int handicap = min(static_cast<int>(data->Set.size()), stats.epochCount);
        newEps = (.1 +greedyEps*exp(-handicap/100.));//*agentId/Real(agentId+1);
    }
    uniform_real_distribution<Real> dis(0.,1.);
    //if(dis(*gen) < newEps) a.getRand();
    if(dis(*gen) < newEps) {
        const int randomActionLabel = nOutputs*dis(*gen);
        a.unpack(randomActionLabel);
        //printf("Random action %d %d %f %d %f\n",data->Set.size(),stats.epochCount,newEps, a.vals[0], a.valsContinuous[0]);
    }
    
    //if (info!=1) printf("Agent %d: %s > %s with %s rewarded with %f acting %s\n", agentId, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(), r ,a.print().c_str());
}

void NFQ::Train_BPTT(const int seq, const int first, const int thrID)
{
    if(not net->allocatedFrozenWeights) die("Gitouttahier!\n");
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs), errs(nOutputs, 0);
    const int ndata = data->Set[seq]->tuples.size();
    
    //first prediction in sequence without recurrent connections
    net->predict(data->Set[seq]->tuples[0]->s, Qhats, net->series[first]);
    
    for (int k=0; k<ndata-1; k++) {//state in k=[0:N-2], act&rew in k+1
        //Q(sNew) predicted at previous loop with moving wghts is current Q
        Qs = Qhats;
        //this tuple contains a, sNew, reward:
        const Tuple * const _t = data->Set[seq]->tuples[k+1];
        
        const bool terminal = k+2==ndata && data->Set[seq]->ended;
        if (not terminal) {
            net->predict(_t->s, Qhats,   net->series[first+k], net->series[first+k+1]);
            net->predict(_t->s, Qtildes, net->series[first+k], net->series[first+ndata],
                         net->tgt_weights,  net->tgt_biases);
        }
        
        // find best action for sNew with moving wghts, evaluate it with tgt wgths:
        // Double Q Learning ( http://arxiv.org/abs/1509.06461 )
        int Nbest;
        Real Vhat(-1e10);
        for (int i=0; i<nOutputs; i++) {
            errs[i] = 0.;
            if(Qhats[i]>Vhat) {
                Vhat=Qhats[i];
                Nbest=i;
            }
        }
        const Real target = (terminal) ? _t->r : _t->r + gamma*Qtildes[Nbest];
        //printf("target %f rew %f %d %f\n",target, _t->r, _t->a, _t->aC[0]);
        const Real err =  (target - Qs[_t->a]);
        errs[_t->a] = err;
        net->setOutputErrors(errs, net->series[first+k]);
        //*(net->series[first+k]->errvals +net->iOutputs+_t->a) = err;
        
        dumpStats(Vstats[thrID], Qs[_t->a], err, Qs);
    }
    
    net->computeDeltasSeries(net->series, first, first+ndata-2);

    if (first==0)
        net->computeAddGradsSeries(net->series, 0, ndata-2, net->grad);
    else
        net->computeAddGradsSeries(net->series, first, first+ndata-2, net->Vgrad[thrID]);
}

void NFQ::Train(const int seq, const int samp, const int first, const int thrID)
{
    if(not net->allocatedFrozenWeights) die("Allocate them!\n");
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs), errs(nOutputs, 0);
    const int ndata = data->Set[seq]->tuples.size();

    const Tuple * const _t = data->Set[seq]->tuples[samp+1];
    net->predict(data->Set[seq]->tuples[samp]->s, Qs, net->series[first]);
    
    const bool term = samp+2==ndata && data->Set[seq]->ended;
    if (not term) {
        net->predict(_t->s, Qhats,   net->series[first+1]);
        net->predict(_t->s, Qtildes, net->series[first+1], net->tgt_weights, net->tgt_biases);
    }
    
    // find best action for sNew with moving wghts, evaluate it with tgt wgths:
    // Double Q Learning ( http://arxiv.org/abs/1509.06461 )
    int Nbest;
    Real Vhat(-1e10);
    for (int i=0; i<nOutputs; i++) {
        if(Qhats[i]>Vhat) {
            Vhat=Qhats[i];
            Nbest=i;
        }
    }
    
    const Real target = (term) ? _t->r : _t->r + gamma*Qtildes[Nbest];
    const Real err =  (target - Qs[_t->a]);
    errs[_t->a] = err;
    net->setOutputErrors(errs, net->series[first]);
    //*(net->series[first]->errvals +net->iOutputs+_t->a) = err;
    
    dumpStats(Vstats[thrID], Qs[_t->a], err, Qs);
    net->computeDeltas(net->series[first]);
    
    if (first==0) net->computeAddGrads(net->series[first], net->grad);
    else          net->computeAddGrads(net->series[first], net->Vgrad[thrID]);
    
}