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

NFQ::NFQ(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner(comm,_env,settings)
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

void NFQ::select(const int agentId, State& s, Action& a, State& sOld,
									Action& aOld, const int info, Real r)
{
		if (info!=1)
		data->passData(agentId, info, sOld, aOld, s, r);  //store sOld, aOld -> sNew, r
		if (info == 2) return;
		assert(info==1 || data->Tmp[agentId]->tuples.size());
    Activation* currActivation = net->allocateActivation();
    vector<Real> output(nOutputs);

    if (info==1) {// if new sequence, sold, aold and reward are meaningless
				vector<Real> inputs(nInputs,0);
		    s.copy_observed(inputs);
    		vector<Real> scaledSold = data->standardize(inputs);
        net->predict(scaledSold, output, currActivation);
		} else {   //then if i'm using RNN i need to load recurrent connections
				const Tuple* const last = data->Tmp[agentId]->tuples.back();
				vector<Real> scaledSold = data->standardize(last->s);
        Activation* prevActivation = net->allocateActivation();
        net->loadMemory(net->mem[agentId], prevActivation);
        net->predict(scaledSold, output, prevActivation, currActivation);
        _dispose_object(prevActivation);
    }

    //save network transition
    net->loadMemory(net->mem[agentId], currActivation);
    _dispose_object(currActivation);

    //load computed policy into a
		const int indBest = maxInd(output);
    a.set(indBest);

    //random action?
    Real newEps(greedyEps);
    if (bTrain) { //if training: anneal random chance if i'm just starting to learn
			const double handicap = min(data->Set.size()/1e2, opt->nepoch/epsAnneal);
      newEps = exp(-handicap) + greedyEps;//*agentId/Real(agentId+1);
    }
    uniform_real_distribution<Real> dis(0.,1.);

    if(dis(*gen) < newEps) a.set(nOutputs*dis(*gen));

    #ifdef _dumpNet_
		if (!bTrain) dumpNetworkInfo(agentId);
    #endif
}

void NFQ::dumpNetworkInfo(const int agentId)
{
	net->dump(agentId);

 	const int ndata = data->Tmp[agentId]->tuples.size(); //last one already placed
	if (ndata == 0) return;

	vector<Real> Qs(nOutputs);
	vector<Activation*> timeSeries_base = net->allocateUnrolledActivations(ndata);
	net->clearErrors(timeSeries_base);

	for (int k=0; k<ndata; k++) {
		const Tuple * const _t = data->Tmp[agentId]->tuples[k];
		vector<Real> scaledSnew = data->standardize(_t->s);
		net->predict(scaledSnew, Qs, timeSeries_base, k);
	}

	const int thisAction = aInfo.actionToLabel(data->Tmp[agentId]->tuples[ndata-1]->a);
	//sensitivity of value for this action in this state wrt all previous inputs
	for (int ii=0; ii<ndata; ii++)
	for (int i=0; i<nInputs; i++) {
		 vector<Activation*> timeSeries_diff = net->allocateUnrolledActivations(ndata);

		for (int k=0; k<ndata; k++) {
			 const Tuple * const _t = data->Tmp[agentId]->tuples[k];
			vector<Real> scaledSnew = data->standardize(_t->s);
			if (k==ii) scaledSnew[i] = 0;
			net->predict(scaledSnew, Qs, timeSeries_diff, k);
		}

		vector<Real> out_diff = net->getOutputs(timeSeries_diff.back());
		vector<Real> out_base = net->getOutputs(timeSeries_base.back());
		 const Tuple * const _t = data->Tmp[agentId]->tuples[ii];
		vector<Real> scaledSnew = data->standardize(_t->s);
		timeSeries_base[ii]->errvals[i] = (out_diff[thisAction]-out_base[thisAction])/scaledSnew[i];

		net->deallocateUnrolledActivations(&timeSeries_diff);
	}

	string fname="gradInputs_"+to_string(agentId)+"_"+to_string(ndata)+".dat";
	ofstream out(fname.c_str());
	if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
	for (int k=0; k<ndata; k++) {
		for (int j=0; j<nInputs; j++)
			out << timeSeries_base[k]->errvals[j] << " ";
		out << "\n";
	}
	out.close();

 net->deallocateUnrolledActivations(&timeSeries_base);
}

void NFQ::Train_BPTT(const int seq, const int thrID) const
{
    assert(net->allocatedFrozenWeights);
    vector<Real> Qs(nOutputs),Qhats(nOutputs),Qtildes(nOutputs),errs(nOutputs);
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
    			net->predict(scaledSnew, Qtildes, timeSeries[k], tgtActivation,
																			net->tgt_weights, net->tgt_biases);

          if (k+2==ndata)
						net->predict(scaledSnew, Qhats, timeSeries[k], tgtActivation);
          else  //used for next transition:
						net->predict(scaledSnew, Qhats, timeSeries, k+1);
        }

        // find best action for sNew with moving wghts, evaluate it with tgt wgths:
        // Double Q Learning ( http://arxiv.org/abs/1509.06461 )
        const int indBest = maxInd(Qhats);
        for (int i=0; i<nOutputs; i++) errs[i] = 0.;

        const Real target = (terminal) ? _t->r : _t->r + gamma*Qtildes[indBest];
        const int action = aInfo.actionToLabel(_t->a);
        const Real err =  (target - Qs[action]);
        //printf("t %f r %f e %f Q %f\n", target, _t->r, err, Qs[action]); fflush(0);
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
    vector<Real> Qs(nOutputs),Qhats(nOutputs),Qtildes(nOutputs),errs(nOutputs);

    vector<Real> scaledSold =data->standardize(data->Set[seq]->tuples[samp]->s);
    const Tuple* const _t = data->Set[seq]->tuples[samp+1];
    Activation* sOldActivation = net->allocateActivation();
    sOldActivation->clearErrors();

    net->predict(scaledSold, Qs, sOldActivation);

    const bool terminal = samp+2==ndata && data->Set[seq]->ended;
    if (not terminal) {
    	vector<Real> scaledSnew = data->standardize(_t->s);
        Activation* sNewActivation = net->allocateActivation();
        net->predict(scaledSnew, Qhats,   sNewActivation);
        net->predict(scaledSnew, Qtildes, sNewActivation,
															net->tgt_weights, net->tgt_biases);
        _dispose_object(sNewActivation);
    }

    // find best action for sNew with moving wghts, evaluate it with tgt wgths:
    // Double Q Learning ( http://arxiv.org/abs/1509.06461 )
		const int indBest = maxInd(Qhats);
		for (int i=0; i<nOutputs; i++) errs[i] = 0.;

    const Real target = (terminal) ? _t->r : _t->r + gamma*Qtildes[indBest];
    const int action = aInfo.actionToLabel(_t->a);
    const Real err =  (target - Qs[action]);
    //printf("t %f r %f e %f Q %f\n", target, _t->r, err, Qs[action]); fflush(0);
    errs[action] = err;

    dumpStats(Vstats[thrID], Qs[action], err, Qs);
    if(thrID == 1) net->updateRunning(sOldActivation);
    data->Set[seq]->tuples[samp]->SquaredError = err*err;

    if (thrID==0) net->backProp(errs, sOldActivation, net->grad);
		else net->backProp(errs, sOldActivation, net->Vgrad[thrID]);

    _dispose_object(sOldActivation);
}
