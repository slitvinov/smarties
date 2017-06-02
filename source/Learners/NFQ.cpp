/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "NFQ.h"
#include "../Math/Utils.h"

NFQ::NFQ(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm,_env,settings,settings.nnOutputs)
{
	#ifdef NDEBUG
	if(bRecurrent) die("NFQ recurrent not tested!\n");
	#endif
	buildNetwork(net, opt, vector<Uint>(1,nOutputs), settings);
}

void NFQ::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	vector<Real> output = output_value_iteration(agentId, s, a, sOld, aOld, info, r);
	if(!output.size()) {
		assert(info==2);
		return;
	}
	//load computed policy into a
	const Uint indBest = maxInd(output);
	a.set(indBest);

	//random action?
	const Real anneal = annealingFactor();
	const Real annealedEps = bTrain ? anneal + (1-anneal)*greedyEps : greedyEps;
	uniform_real_distribution<Real> dis(0.,1.);
	if(dis(*gen) < annealedEps) a.set(nOutputs*dis(*gen));

	dumpNetworkInfo(agentId);
}

void NFQ::Train_BPTT(const Uint seq, const Uint thrID) const
{
	const Real rGamma = annealedGamma();
	assert(net->allocatedFrozenWeights && bTrain);
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Real> Qs(nOutputs),Qhats(nOutputs),Qtildes(nOutputs),errs(nOutputs);
	vector<Activation*> timeSeries = net->allocateUnrolledActivations(ndata-1);
	Activation* tgtAct = net->allocateActivation();

	{   //first prediction in sequence without recurrent connections
		const Tuple * const _t = data->Set[seq]->tuples[0];
		net->predict(data->standardize(_t->s), Qhats, timeSeries, 0);
	}

	for (Uint k=0; k<ndata-1; k++)
	{ //state in k=[0:N-2], act&rew in k+1
		Qs = Qhats; //Q(sNew) predicted at previous loop with moving wghts is current Q
		//this tuple contains a, sNew, reward:
		const Tuple * const _t = data->Set[seq]->tuples[k+1];
		const bool terminal = k+2==ndata && data->Set[seq]->ended;
		if (not terminal)
		{
			const vector<Real> snew = data->standardize(_t->s);
			net->predict(snew, Qtildes, timeSeries[k], tgtAct, net->tgt_weights, net->tgt_biases);

			if (k+2==ndata)
				net->predict(snew, Qhats, timeSeries[k], tgtAct);
			else  //used for next transition:
				net->predict(snew, Qhats, timeSeries, k+1);
		}

		// find best action for sNew with moving wghts, evaluate it with tgt wgths:
		// Double Q Learning ( http://arxiv.org/abs/1509.06461 )
		const Uint indBest = maxInd(Qhats);
		const Real target = (terminal) ? _t->r : _t->r + rGamma*Qtildes[indBest];
		const Uint action = aInfo.actionToLabel(_t->a);
		const Real err =  (target - Qs[action]);
		for (Uint i=0; i<nOutputs; i++) errs[i] = i==action ? err : 0;

		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], errs);
		clip_gradient(errs, stdGrad[0]);
		dumpStats(Vstats[thrID], Qs[action], err);
		data->Set[seq]->tuples[k]->SquaredError = err*err;
		net->setOutputDeltas(errs, timeSeries[k]);
	}

	if (thrID==0) net->backProp(timeSeries, net->grad);
	else net->backProp(timeSeries, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&timeSeries);
	_dispose_object(tgtAct);
}

void NFQ::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	const Real rGamma = annealedGamma();
	assert(net->allocatedFrozenWeights && bTrain);
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Real> Qs(nOutputs),Qhats(nOutputs),Qtildes(nOutputs),errs(nOutputs);
	Activation* sOldActivation = net->allocateActivation();

	const Tuple* const t_ = data->Set[seq]->tuples[samp];
	const Tuple* const _t = data->Set[seq]->tuples[samp+1];
	const vector<Real> sold = data->standardize(t_->s);
	net->predict(sold, Qs, sOldActivation);

	const bool terminal = samp+2==ndata && data->Set[seq]->ended;
	if (not terminal)
	{
		const vector<Real> snew = data->standardize(_t->s);
		//vector<Real> scaledSnew = data->standardize(_t->s, __NOISE, thrID);
		Activation* tgtAct = net->allocateActivation();
		net->predict(snew, Qhats,   tgtAct);
		net->predict(snew, Qtildes, tgtAct, net->tgt_weights, net->tgt_biases);
		_dispose_object(tgtAct);
	}

	// find best action for sNew with moving wghts, evaluate it with tgt wgths:
	// Double Q Learning ( http://arxiv.org/abs/1509.06461 )
	const Uint indHats = maxInd(Qhats);
	const Uint indTild = maxInd(Qtildes);
	const Real vNext = rGamma * minAbsValue(Qtildes[indHats], Qhats[indTild]);
	const Real target =(terminal)? _t->r : _t->r + vNext;
	const Uint action = aInfo.actionToLabel(_t->a);
	const Real err =  (target - Qs[action]);
	for (Uint i=0; i<nOutputs; i++) errs[i] = i==action ? err : 0;

	statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], errs);
	clip_gradient(errs, stdGrad[0]);
	dumpStats(Vstats[thrID], Qs[action], err);
	data->Set[seq]->tuples[samp]->SquaredError = err*err;

	if (thrID==0) net->backProp(errs, sOldActivation, net->grad);
	else net->backProp(errs, sOldActivation, net->Vgrad[thrID]);
	_dispose_object(sOldActivation);
}
