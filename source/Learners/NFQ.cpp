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

    net = new Network(settings);
    //check if environment wants a particular network structure
	if (not env->predefinedNetwork(net))
	{ //if that was true, environment created the layers it wanted, else we read the settings:
		net->addInput(nInputs);
		for (int i=0; i<lsize.size(); i++) net->addLayer(lsize[i], lType);
		net->addOutput(nOutputs, "Normal");
	}
    net->build();
    //opt = new Optimizer(net, profiler, settings);
    opt = new AdamOptimizer(net, profiler, settings);
}

void NFQ::select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r)
{
    Activation* currActivation = net->allocateActivation();
    vector<Real> output(nOutputs), inputs(nInputs);
    s.copy_observed(inputs);
    vector<Real> scaledSold = data->standardize(inputs);

    if (info==1)// if new sequence, sold, aold and reward are meaningless
        net->predict(scaledSold, output, currActivation);
    else {   //then if i'm using RNN i need to load recurrent connections
        Activation* prevActivation = net->allocateActivation();
        net->loadMemory(net->mem[agentId], prevActivation);
        net->predict(scaledSold, output, prevActivation, currActivation);
        //also, store sOld, aOld -> sNew, r
        data->passData(agentId, info, sOld, aOld, s, r);
        _dispose_object(prevActivation);
    }
    
    //save network transition
    net->loadMemory(net->mem[agentId], currActivation);
    _dispose_object(currActivation);

    #ifdef _dumpNet_
    net->dump(agentId);
    #endif
    
    //load computed policy into a
    Real Val(-1e6); int Nbest;
    for (int i=0; i<nOutputs; ++i) {
        if (output[i]>Val) { Nbest=i; Val=output[i]; }
    }
    a.set(aInfo.labelToAction(Nbest));
    
    //random action?
    Real newEps(greedyEps);
    if (bTrain) { //if training: anneal random chance if i'm just starting to learn
        const int handicap = min(static_cast<int>(data->Set.size())/500., stats.epochCount/10.);
        newEps = exp(-handicap) + greedyEps;//*agentId/Real(agentId+1);
    }
    uniform_real_distribution<Real> dis(0.,1.);
    
    if(dis(*gen) < newEps) a.set(aInfo.labelToAction(nOutputs*dis(*gen)));
      /*
    if(dis(*gen) < newEps) {
        a.set(aInfo.labelToAction(nOutputs*dis(*gen)));
        printf("Random action %f  for state  %s\n", 
         a.vals[0], s.print().c_str()); fflush(0);
    } else {
        printf("Net selected %d %f for state %s\n",
         Nbest, a.vals[0], s.print().c_str()); fflush(0);
    }
      */
}

void NFQ::Train_BPTT(const int seq, const int thrID) const
{
    assert(net->allocatedFrozenWeights);
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs), errs(nOutputs, 0);
    const int ndata = data->Set[seq]->tuples.size();
    vector<Activation*> timeSeries = net->allocateUnrolledActivations(ndata-1);
    Activation* tgtActivation = net->allocateActivation();
    net->clearErrors(timeSeries);

    {   //first prediction in sequence without recurrent connections
		vector<Real> scaledSold = data->standardize(data->Set[seq]->tuples[0]->s);
		net->predict(scaledSold, Qhats, timeSeries, 0);
    }

    for (int k=0; k<ndata-1; k++) { //state in k=[0:N-2], act&rew in k+1
        Qs = Qhats; //Q(sNew) predicted at previous loop with moving wghts is current Q
        
        const Tuple * const _t = data->Set[seq]->tuples[k+1]; //this tuple contains a, sNew, reward
        const bool terminal = k+2==ndata && data->Set[seq]->ended;

        if (not terminal) {
        	vector<Real> scaledSnew = data->standardize(_t->s);
    		net->predict(scaledSnew, Qtildes, timeSeries[k], tgtActivation, net->tgt_weights, net->tgt_biases);

            if (k+2==ndata) net->predict(scaledSnew, Qhats, timeSeries[k], tgtActivation);
            else            net->predict(scaledSnew, Qhats, timeSeries, k+1);
        }
        
        // find best action for sNew with moving wghts, evaluate it with tgt wgths:
        // Double Q Learning ( http://arxiv.org/abs/1509.06461 )
        int Nbest(-1);
        Real Vhat(-1e10);
        for (int i=0; i<nOutputs; i++) {
            errs[i] = 0.;
            if(Qhats[i]>Vhat) { Vhat=Qhats[i]; Nbest=i;  }
        }
        const Real target = (terminal) ? _t->r : _t->r + gamma*Qtildes[Nbest];
        const int action = aInfo.actionToLabel(_t->a);
        const Real err =  (target - Qs[action]);
        errs[action] = err;
        net->setOutputDeltas(errs, timeSeries[k]);
        dumpStats(Vstats[thrID], Qs[action], err, Qs);
        data->Set[seq]->tuples[k]->SquaredError = err*err;
        if(thrID == 1) net->updateRunning(timeSeries[k]);
    }

    if (thrID==0) net->backProp(timeSeries, net->grad);
    else net->backProp(timeSeries, net->Vgrad[thrID]);
    net->deallocateUnrolledActivations(&timeSeries);
    _dispose_object(tgtActivation);
}

void NFQ::Train(const int seq, const int samp, const int thrID) const
{
    assert(net->allocatedFrozenWeights);
    const int ndata = data->Set[seq]->tuples.size();
    vector<Real> Qs(nOutputs), Qhats(nOutputs), Qtildes(nOutputs), errs(nOutputs, 0);

    vector<Real> scaledSold = data->standardize(data->Set[seq]->tuples[samp]->s);
    const Tuple* const _t = data->Set[seq]->tuples[samp+1];
    Activation* sOldActivation = net->allocateActivation();
    sOldActivation->clearErrors();

    net->predict(scaledSold, Qs, sOldActivation);

    const bool terminal = samp+2==ndata && data->Set[seq]->ended;
    if (not terminal) {
    	vector<Real> scaledSnew = data->standardize(_t->s);
        Activation* sNewActivation = net->allocateActivation();
        net->predict(scaledSnew, Qhats,   sNewActivation);
        net->predict(scaledSnew, Qtildes, sNewActivation, net->tgt_weights, net->tgt_biases);
        _dispose_object(sNewActivation);
    }
    
    // find best action for sNew with moving wghts, evaluate it with tgt wgths:
    // Double Q Learning ( http://arxiv.org/abs/1509.06461 )
    int Nbest;
    Real Vhat(-1e10);
    for (int i=0; i<nOutputs; i++) {
    	if(Qhats[i]>Vhat) { Vhat=Qhats[i]; Nbest=i;  }
    }
    
    const Real target = (terminal) ? _t->r : _t->r + gamma*Qtildes[Nbest];
    const int action = aInfo.actionToLabel(_t->a);
    const Real err =  (target - Qs[action]);
    errs[action] = err;
    
    dumpStats(Vstats[thrID], Qs[action], err, Qs);
    if(thrID == 1) net->updateRunning(sOldActivation);
    data->Set[seq]->tuples[samp]->SquaredError = err*err;

    if (thrID==0) net->backProp(errs, sOldActivation, net->grad);
	else net->backProp(errs, sOldActivation, net->Vgrad[thrID]);

    _dispose_object(sOldActivation);
}
