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
	die("DPG with BPTT not implemented: do eet!\n");
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
        
      //now i need to compute dQ/dA, for Q net use tgt weight throughout
        gradient[0] = 1.;
        net->setOutputErrors(gradient, net->series[first]);
        net->computeDeltas(net->series[first]
                           ,net->tgt_weights,net->tgt_biases
                           );
        
        //also compute deltas on input layer
        vector<Real> QinputGrad(nS+nA);
        net->computeDeltasInputs(QinputGrad,net->series[first]
                                 ,net->tgt_weights,net->tgt_biases
                                 );
        
        //take out the components relevant to actions
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
