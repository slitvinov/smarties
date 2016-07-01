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
    Real newEps(greedyEps);
    vector<Real> output(nOutputs), inputs(nInputs);
    const int handicap = min(static_cast<int>(T->Set.size()), stats.epochCount*10);
    if (bTrain) newEps = (.1 +greedyEps*exp(-handicap/500.));//*agentId/Real(agentId+1);
    if (info!=1) T->passData(agentId, info, sOld, aOld, s, r);
    
    s.scaleUsed(inputs);
    net->expandMemory(net->mem[agentId], net->series[0]); //RNN to update recurrent signals
    net->predict(inputs, output, net->series[0], net->series[1]);
    #ifdef _dumpNet_
    net->dump(agentId);
    #endif
    net->expandMemory(net->mem[agentId], net->series[1]);
    
    Real Val(-1e6); int Nbest;
    for (int i=0; i<nOutputs; ++i) {
        if (output[i]>Val) { Nbest=i; Val=output[i]; }
    }
    a.unpack(Nbest);
    //printf("%d %d %f\n",a.vals[0],i,a.valsContinuous[0]);
    
    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(*gen) < newEps) a.getRand();
}

void NFQ::Train(const int thrID, const int seq, const int first)
{
    if(not net->allocatedFrozenWeights) die("Gitouttahier!\n");
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs);

    net->predict(T->Set[seq]->tuples[0]->s, Qhats, net->series[first]);
    const int ndata = T->Set[seq]->tuples.size();
    
    for (int k=0; k<ndata-1; k++) {//state in k=[0:N-2], act&rew in k+1
        Qs = Qhats;
        const Tuple * const _t = T->Set[seq]->tuples[k+1];
        
        if (k+2==ndata && T->Set[seq]->ended) {
            for (int i=0; i<nOutputs; i++) {
                *(net->series[first+k]->errvals +net->iOutputs+i) = 0;
            }
            const Real target = _t->r;
            const Real err =  (target - Qs[_t->a]);
            dumpStats(Vstats[thrID], target, err, Qs);
            *(net->series[first+k]->errvals +net->iOutputs+_t->a) = err;
        } else {
            net->predict(_t->s, Qhats,   net->series[k], net->series[first+1]);
            net->predict(_t->s, Qtildes, net->series[k], net->series[first+ndata],
                                    net->frozen_weights, net->frozen_biases);
            int Nbest; Real Vhat(-1e10);
            for (int i=0; i<nOutputs; i++) {
                *(net->series[first+k]->errvals +net->iOutputs +i) = 0;
                if (Qhats[i]>Vhat)  { Nbest=i; Vhat=Qhats[i]; }
            }
            const Real target = _t->r + gamma*Qtildes[Nbest];
            const Real err =  (target - Qs[_t->a]);
            dumpStats(Vstats[thrID], target, err, Qs);
            *(net->series[first+k]->errvals +net->iOutputs+_t->a) = err;
        }
    }
    {
        net->computeDeltasSeries(net->series, first, first+ndata-2);
        
        for (int k=first; k<first+ndata-1; k++)
            net->computeAddGradsSeries(net->series, k, net->Vgrad[thrID]);
    }
}

void NFQ::Train(const vector<int>& seq)
{
    if(not net->allocatedFrozenWeights) die("Gitouttahier!\n");
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs);
    int countUpdate(0);

    for (int jnd(0); jnd<seq.size(); jnd++) {
        const int ind = seq[jnd];
        const int ndata = T->Set[ind]->tuples.size();
        net->allocateSeries(ndata);
        //net->assignDropoutMask();
        net->predict(T->Set[ind]->tuples[0]->s, Qhats, net->series[0]);
        
        for (int k=0; k<ndata-1; k++) {//state in k=[0:N-1], act&rew in k+1
            Qs = Qhats;
            const Tuple * const _t = T->Set[ind]->tuples[k+1];
            
            if (k+2==ndata && T->Set[ind]->ended) {
                for (int i=0; i<nOutputs; i++) {
                    *(net->series[k]->errvals +net->iOutputs+i) = 0;
                }
                const Real target = _t->r;
                const Real err =  (target - Qs[_t->a]);
                dumpStats(target, err, Qs);
                *(net->series[k]->errvals +net->iOutputs+_t->a) = err;
            } else {
                net->predict(_t->s, Qhats,   net->series[k], net->series[k+1]);
                net->predict(_t->s, Qtildes, net->series[k], net->series[ndata],
                                        net->frozen_weights, net->frozen_biases);
                int Nbest; Real Vhat(-1e10);
                for (int i=0; i<nOutputs; i++) {
                    *(net->series[k]->errvals +net->iOutputs +i) = 0;
                    if (Qhats[i]>Vhat)  { Nbest=i; Vhat=Qhats[i]; }
                }
                const Real target = _t->r + gamma*Qtildes[Nbest];
                const Real err =  (target - Qs[_t->a]);
                dumpStats(target, err, Qs);
                *(net->series[k]->errvals +net->iOutputs+_t->a) = err;
            }
        }
        {
            net->computeDeltasSeries(net->series, 0, ndata-2);
            
            for (int k=0; k<ndata-1; k++) {
                net->computeGradsSeries(net->series, k, net->_grad);
                opt->stackGrads(net->grad,net->_grad);
                countUpdate++;
            }
            //net->removeDropoutMask(); //before the update!!!
        }
    }
    opt->nepoch=stats.epochCount;
    opt->update(net->grad, countUpdate);
}

void NFQ::Train(const int thrID, const int seq, const int samp, const int first)
{
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs);

    const Tuple * const _t = T->Set[seq]->tuples[samp+1];
    net->predict(T->Set[seq]->tuples[samp]->s, Qs, net->series[first]);
    
    const bool term = samp+2==T->Set[seq]->tuples.size() && T->Set[seq]->ended;
    if (not term) {
        net->predict(_t->s, Qhats,   net->series[first], net->series[first+1]);
        net->predict(_t->s, Qtildes, net->series[first], net->series[first+2],
                                net->frozen_weights, net->frozen_biases);
    }
    int Nbest; Real Vhat(-1e10);
    for (int i=0; i<nOutputs; i++) {
        *(net->series[first]->errvals +net->iOutputs+i) = 0;
        if (Qhats[i]>Vhat)  { Nbest=i; Vhat=Qhats[i]; }
    }
    
    const Real target = (term) ? _t->r : _t->r + gamma*Qtildes[Nbest];
    const Real err =  (target - Qs[_t->a]);
    dumpStats(Vstats[thrID], target, err, Qs);
    *(net->series[first]->errvals +net->iOutputs+_t->a) = err;
    net->computeDeltas(net->series[first]);
    net->computeAddGrads(net->series[first], net->Vgrad[thrID]);
}

void NFQ::Train(const vector<int>& seq, const vector<int>& samp)
{
    if(not net->allocatedFrozenWeights) die("Allocate them!\n");
    const int ndata = seq.size();
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs);
    int countUpdate(0);
    
    for (int k=0; k<ndata; k++) { //TODO clean this shit up
        const int knd(seq[k]), ind(samp[k]);
        net->predict(T->Set[knd]->tuples[ind]->s, Qs, net->series[0]);
        const Tuple * const _t = T->Set[knd]->tuples[ind+1];
        
        const bool term = ind+2==T->Set[knd]->tuples.size() && T->Set[knd]->ended;
        if(not term) {
            net->predict(_t->s, Qhats,   net->series[0], net->series[1]);
            net->predict(_t->s, Qtildes, net->series[0], net->series[2],
                                    net->frozen_weights, net->frozen_biases);
        }
        int Nbest; Real Vhat(-1e10);
        for (int i=0; i<nOutputs; i++) {
            *(net->series[0]->errvals +net->iOutputs+i) = 0;
            if (Qhats[i]>Vhat)  { Nbest=i; Vhat=Qhats[i]; }
        }
        
        const Real target = (term) ? _t->r : _t->r + gamma*Qtildes[Nbest];
        const Real err =  (target - Qs[_t->a]);
        dumpStats(target, err, Qs);
        *(net->series[0]->errvals +net->iOutputs+_t->a) = err;
        
        net->computeDeltas(net->series[0]);
        net->computeGrads(net->series[0], net->_grad);
        opt->stackGrads(net->grad,net->_grad);
        countUpdate++;
    }
    opt->nepoch=stats.epochCount;
    opt->update(net->grad, countUpdate);
}
