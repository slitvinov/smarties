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
	string lType = bRecurrent ? "LSTM" : "Normal";
	vector<int> lsize;
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
	if (env->predefinedNetwork(net))
		die("Predefined env structure still unsupported for DPG.\n");

	net = new Network(settings);
	net->addInput(nS+nA);
	for (int i=0; i<lsize.size(); i++) net->addLayer(lsize[i], lType);
	net->addOutput(1, "Normal");
	net->build();
	opt = new AdamOptimizer(net, profiler, settings);

	net_policy = new Network(settings);
	net_policy->addInput(nS);
	for (int i=0; i<lsize.size(); i++) net_policy->addLayer(lsize[i], lType);
	net_policy->addOutput(nA, "Normal");
	net_policy->build();
	opt_policy = new AdamOptimizer(net_policy, profiler, settings);
}

void DPG::select(const int agentId,State& s,Action& a,State& sOld,Action& aOld,const int info,Real r)
{
    Activation* currActivation = net_policy->allocateActivation();
    vector<Real> output(nA), inputs(nS);
    s.scaleUsed(inputs);
    
    if (info==1) {// if new sequence, sold, aold and reward are meaningless
        net_policy->predict(inputs, output, currActivation);
    } else { //then if i'm using RNN i need to load recurrent connections
        Activation* prevActivation = net_policy->allocateActivation();
        net_policy->loadMemory(net_policy->mem[agentId], prevActivation);
        net_policy->predict(inputs, output, prevActivation, currActivation);
        data->passData(agentId, info, sOld, aOld, s, r);  //store sOld, aOld -> sNew, r
        _dispose_object(prevActivation);
    }
#ifdef _dumpNet_
    net_policy->dump(agentId);
#endif
    //save network transition
    net_policy->loadMemory(net_policy->mem[agentId], currActivation);
    _dispose_object(currActivation);
    //load computed policy into a
    a.set_fromScaled(output);
    
    Real newEps(greedyEps); //random action?
    if (bTrain) { //if training: anneal random chance if i'm just starting to learn
        const int handicap = min(static_cast<int>(data->Set.size())/500., stats.epochCount/10.);
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

void DPG::Train_BPTT(const int seq, const int thrID)
{
	die("DPG with BPTT not implemented: do eet!\n");
}

void DPG::Train(const int seq, const int samp, const int thrID)
{
    assert(net->allocatedFrozenWeights && net_policy->allocatedFrozenWeights);
    
    //this tuple contains a, sNew, reward:
    const Tuple * const _t = data->Set[seq]->tuples[samp+1];
    //sOld contained in previous tuple
    const Tuple * const _tOld = data->Set[seq]->tuples[samp];
    Activation* sOldAAct = net_policy->allocateActivation();
    Activation* sNewAAct = net_policy->allocateActivation();
    Activation* sOldQAct = net->allocateActivation();
    Activation* sNewQAct = net->allocateActivation();
    sOldAAct->clearErrors();
    sOldQAct->clearErrors();
    sNewQAct->clearErrors();
    
    //update Q network:
    vector<Real> vSnew(1), Q(1), gradient(1);
    { //join state and action to predict Q
        vector<Real> input(_tOld->s);
        input.insert(input.end(),_t->aC.begin(),_t->aC.end());
        net->predict(input, Q, sOldQAct);
    }

    const bool terminal = samp+2==data->Set[seq]->tuples.size() && data->Set[seq]->ended;
    if (not terminal) {
        //first predict best action with policy NN w/ target weights
        vector<Real> policy(nA);
        net_policy->predict(_t->s, policy, sNewAAct, net_policy->tgt_weights, net_policy->tgt_biases);
        //then predict target value for V(s_new)
        vector<Real> input(_t->s);
        input.insert(input.end(),policy.begin(),policy.end());
        net->predict(input, vSnew, sNewQAct, net->tgt_weights, net->tgt_biases);
    }
    
    const Real target = (terminal) ? _t->r : _t->r + gamma*vSnew[0];
    gradient[0] = target - Q[0];
    
    if (thrID==0) net->backProp(gradient, sOldQAct, net->grad);
    else net->backProp(gradient, sOldQAct, net->Vgrad[thrID]);
    dumpStats(Vstats[thrID], Q[0], gradient[0], Q);
    
    //now update policy network:
    { //predict policy for sOld
        vector<Real> policy(nA), pol_gradient(nA);
        net_policy->predict(_tOld->s, policy, sOldAAct);
        
        //use it to compute activation with frozen weitghts for Q net
        vector<Real> input(_tOld->s);
        input.insert(input.end(), policy.begin(), policy.end());
        net->predict(input, Q, sNewQAct, net->tgt_weights, net->tgt_biases);
        
        //now i need to compute dQ/dA, for Q net use tgt weight throughout
        gradient[0] = 1.;
    	Grads* tmp_grad = new Grads(net->nWeights,net->nBiases);
    	net->backProp(gradient, sNewQAct, net->tgt_weights, tmp_grad);
        _dispose_object(tmp_grad);
    	for(int i=0;i<nA;i++) pol_gradient[i] = sNewQAct->errvals[nS+i];

    	if (thrID==0) net_policy->backProp(pol_gradient, sOldAAct, net_policy->grad);
    	else net_policy->backProp(pol_gradient, sOldAAct, net_policy->Vgrad[thrID]);
    }

    _dispose_object(sOldAAct);
    _dispose_object(sNewAAct);
    _dispose_object(sOldQAct);
    _dispose_object(sNewQAct);
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
    
    opt_policy->stackGrads(net_policy->grad, net_policy->Vgrad); //add up gradients across threads
    opt_policy->update(net_policy->grad,nAddedGradients); //update
}

void DPG::updateNNWeights(const int nAddedGradients)
{
    opt->nepoch=stats.epochCount;  //used to anneal learning rate
    opt->update(net->grad,nAddedGradients);
    
    opt_policy->nepoch=stats.epochCount;  //used to anneal learning rate
    opt_policy->update(net_policy->grad,nAddedGradients);
}