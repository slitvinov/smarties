/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "DPG.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cmath>

DPG::DPG(Environment* env, Settings & settings) : Learner(env,settings), nS(env->sI.dimUsed), nA(env->aI.dim)
{
    vector<int> lsize;
    lsize.push_back(nS);
    lsize.push_back(settings.nnLayer1);
    if (settings.nnLayer2>1) {
        lsize.push_back(settings.nnLayer2);
        if (settings.nnLayer3>1) {
            lsize.push_back(settings.nnLayer3);
            if (settings.nnLayer4>1) {
                lsize.push_back(settings.nnLayer4);
                if (settings.nnLayer5>1) {
                    lsize.push_back(settings.nnLayer5);
                }
            }
        }
    }
    lsize.push_back(nA);
    
    net_policy = new Network(lsize, bRecurrent, settings);
    opt_policy = new AdamOptimizer(net, profiler, settings);
}

void DPG::select(const int agentId,State& s,Action& a,State& sOld,Action& aOld,const int info,Real r)
{
    vector<Real> output(nA), inputs(nS);
    s.scaleUsed(inputs);
    
    if (info==1) //is it the first action performed by agent?
    {   //then old state, old action and reward are meaningless, just give the action
        net_policy->predict(inputs, output, net_policy->series[1]);
    }
    else
    {   //then if i'm using RNN i need to load recurrent connections
        net_policy->expandMemory(net_policy->mem[agentId], net_policy->series[0]);
        net_policy->predict(inputs, output, net_policy->series[0], net_policy->series[1]);
        //also, store sOld, aOld -> sNew, r
        data->passData(agentId, info, sOld, aOld, s, r);
    }
#ifdef _dumpNet_
    net_policy->dump(agentId);
#endif
    
    //save network transition
    net_policy->expandMemory(net_policy->mem[agentId], net_policy->series[1]);
    
    //load computed policy into a
    a.descale(output);
    
    //random action?
    Real newEps(greedyEps);
    if (bTrain) { //if training: anneal random chance if i'm just starting to learn
        const Real crutch_1 = 1e6;//stats.relE < 0.5 ? 100. : 1./stats.relE;
        const Real crutch_2 = static_cast<int>(data->Set.size())/500.;
        const Real crutch_3 = stats.epochCount/10.;
        const Real handicap = min(min(crutch_1,crutch_2),crutch_3);
        newEps = (.1 +greedyEps*exp(-handicap));//*agentId/Real(agentId+1);
        //printf("Random action %f %f %f %f\n",crutch_1,crutch_2,crutch_3,newEps);
    }
    
    uniform_real_distribution<Real> dis(0.,1.);
    if(dis(*gen) < newEps) {
        
        a.getRandom();
        //printf("Random action %d  %f  for state %s %s\n",a.vals[0], a.valsContinuous[0],s.printScaled().c_str(),s.print().c_str());fflush(0);
    } else {
        
        //printf("Net selected %f %f for state %s\n",act[0],a.valsContinuous[0],s.printScaled().c_str());fflush(0);
    }
    
    //if (info!=1) printf("Agent %d: %s > %s with %s rewarded with %f acting %s\n", agentId, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(), r ,a.print().c_str());
}

void DPG::Train_BPTT(const int seq, const int first, const int thrID)
{
    /*
    if(not net->allocatedFrozenWeights) die("Gitouttahier!\n");
    vector<Real> target(nOutputs), output(nOutputs), gradient(nOutputs);
    const int ndata = data->Set[seq]->tuples.size();
    
    for (int k=0; k<ndata-1; k++) {//state in k=[0:N-2], act&rew in k+1, last state (N-1) not used for Q update
        //this tuple contains a, sNew, reward:
        const Tuple * const _t = data->Set[seq]->tuples[k+1];
        //this tuple contains sOld:
        const Tuple * const _tOld = data->Set[seq]->tuples[k];
        //printf("%f %f %f %f %f %f\n",_tOld->s[0],_tOld->s[1], _t->aC[0], _t->s[0],_t->s[1], _t->r);
        //if first in a sequence, no input from recurrent links
        if(k==0)
            net->predict(_tOld->s, output, net->series[first]);
        else
            net->predict(_tOld->s, output, net->series[first+k-1], net->series[first+k]);
        
        const bool terminal = k+2==ndata && data->Set[seq]->ended;
        
        if (not terminal) {
            net->predict(_t->s, target, net->series[first+k], net->series[first+ndata-1],
                         net->tgt_weights,  net->tgt_biases);
        }
        
        Real err = (terminal) ? _t->r : _t->r + gamma*target[0];
        const vector<Real> Q(computeQandGrad(gradient, _t->aC, output, err));
        net->setOutputErrors(gradient, net->series[first+k]);
        //for (int i(0); i<nOutputs; i++) { //put grad into network
        //    *(net->series[first+k]->errvals +net->iOutputs+i) = gradient[i];
        //}
        
        dumpStats(Vstats[thrID], Q[0], err, Q);
    }
    
    net->computeDeltasSeries(net->series, first, first+ndata-2);
    
    if (first==0)
        net->computeAddGradsSeries(net->series, 0, ndata-2, net->grad);
    else
        net->computeAddGradsSeries(net->series, first, first+ndata-2, net->Vgrad[thrID]);
    */
}

void DPG::Train(const int seq, const int samp, const int first, const int thrID)
{
    if(not net->allocatedFrozenWeights) die("Allocate them!\n");
    if(not net_policy->allocatedFrozenWeights) die("Allocate them!\n");
    
    //this tuple contains a, sNew, reward:
    const Tuple * const _t = data->Set[seq]->tuples[samp+1];
    //sOld contained in previous tuple
    const Tuple * const _tOld = data->Set[seq]->tuples[samp];
    
    //update Q network:
    vector<Real> vSnew(1), Q(1), gradient(1);
    const int ndata = data->Set[seq]->tuples.size();

    { //join state and action to predict Q
        vector<Real> input(_tOld->s);
        input.insert(input.end(),_t->aC.begin(),_t->aC.end());
        net->predict(input, Q, net->series[first]);
    }
    
    const bool terminal = samp+2==ndata && data->Set[seq]->ended;
    if (not terminal) {
        //first predict best action with policy NN w/ target weights
        vector<Real> policy(nA);
        net_policy->predict(_t->s, policy, net_policy->series[first+1],net_policy->tgt_weights,net_policy->tgt_biases);
        //then predict target value for V(s_new)
        vector<Real> input(_t->s);
        input.insert(input.end(),policy.begin(),policy.end());
        net->predict(input, vSnew, net->series[first+1],net->tgt_weights,net->tgt_biases);
    }
    
    const Real target = (terminal) ? _t->r : _t->r + gamma*vSnew[0];
    gradient[0] = target - Q[0];
    net->setOutputErrors(gradient, net->series[first]);
    
    dumpStats(Vstats[thrID], Q[0], gradient[0], Q);
    net->computeDeltas(net->series[first]);
    
    if (thrID==0) net->computeAddGrads(net->series[first], net->grad);
    else          net->computeAddGrads(net->series[first], net->Vgrad[thrID]);
    
    //now update policy network:
    
    { //predict policy for sOld
        vector<Real> policy(nA), pol_gradient(nA);
        net_policy->predict(_tOld->s, policy, net_policy->series[first]);
        
      //use it to compute activation with frozen weitghts for Q net
        vector<Real> input(_tOld->s);
        input.insert(input.end(),policy.begin(),policy.end());
        net->predict(input, Q, net->series[first]
                     ,net->tgt_weights,net->tgt_biases
                     );
        
      //now i need to compute dQ/dA, for Q net use tgt weight trughout
        gradient[0] = 1.; //it's like having a smaller learn rate for pol net
        net->setOutputErrors(gradient, net->series[first]);
        net->computeDeltas(net->series[first]
                           ,net->tgt_weights,net->tgt_biases
                           );
        
        //also compute deltas on input layer
        vector<Real> QinputGrad(nS+nA);
        net->computeDeltasInputs(QinputGrad,net->series[first]
                                 ,net->tgt_weights,net->tgt_biases
                                 );
        
        //take out the coponents relevant to actions
        for (int i=0; i<nA; i++) pol_gradient[i] = QinputGrad[i+nS];
        
        net_policy->setOutputErrors(pol_gradient, net_policy->series[first]);

        net_policy->computeDeltas(net_policy->series[first]);
        
        if (thrID==0) net_policy->computeAddGrads(net_policy->series[first], net_policy->grad);
        else          net_policy->computeAddGrads(net_policy->series[first], net_policy->Vgrad[thrID]);
    }
    
}

void DPG::updateTargetNetwork()
{
    if (cntUpdateDelay <= 0) { //DQN-style frozen weight
        #pragma omp master
        cntUpdateDelay = tgtUpdateDelay;
        
        //2 options: either move tgt_wght = (1-a)*tgt_wght + a*wght
        if (tgtUpdateDelay==0) {
            net->moveFrozenWeights(tgtUpdateAlpha);
            net_policy->moveFrozenWeights(tgtUpdateAlpha);
        } else {
            net->updateFrozenWeights(); //or copy tgt_wghts = wghts
            net_policy->updateFrozenWeights(); //or copy tgt_wghts = wghts
        }
    }
    #pragma omp master
    cntUpdateDelay--;
}

void DPG::stackAndUpdateNNWeights(const int nAddedGradients)
{
    opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads
    opt->update(net->grad,nAddedGradients); //update
    
    opt_policy->stackGrads(net->grad, net->Vgrad); //add up gradients across threads
    opt_policy->update(net->grad,nAddedGradients); //update
}

void DPG::updateNNWeights(const int nAddedGradients)
{
    opt->nepoch=stats.epochCount;  //used to anneal learning rate
    opt->update(net->grad,nAddedGradients);
    
    opt_policy->nepoch=stats.epochCount;  //used to anneal learning rate
    opt_policy->update(net->grad,nAddedGradients);
}

void DPG::allocateNNactivations(const int buffer)
{
    net->allocateSeries(buffer);
    net_policy->allocateSeries(buffer);
}
