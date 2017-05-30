/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "NFQ.h"

NFQ::NFQ(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner(comm,_env,settings)
{
	buildNetwork(net, opt, vector<Uint>(1,nOutputs), settings);
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
		prevActivation->loadMemory(net->mem[agentId]);
		net->predict(scaledSold, output, prevActivation, currActivation);
		_dispose_object(prevActivation);
	}

	//save network transition
	currActivation->storeMemory(net->mem[agentId]);
	_dispose_object(currActivation);

	//load computed policy into a
	const Uint indBest = maxInd(output);
	a.set(indBest);

	//random action?
	const Real annealedEps = bTrain ? annealingFactor() + greedyEps : greedyEps;
	uniform_real_distribution<Real> dis(0.,1.);

	if(dis(*gen) < annealedEps) a.set(nOutputs*dis(*gen));

#ifdef _dumpNet_
	if (!bTrain) dumpNetworkInfo(agentId);
#endif
}

void NFQ::dumpNetworkInfo(const int agentId)
{
	net->dump(agentId);

	const Uint ndata = data->Tmp[agentId]->tuples.size(); //last one already placed
	if (ndata == 0) return;

	vector<Real> Qs(nOutputs);
	vector<Activation*> timeSeries_base = net->allocateUnrolledActivations(ndata);
	net->clearErrors(timeSeries_base);

	for (Uint k=0; k<ndata; k++) {
		const Tuple * const _t = data->Tmp[agentId]->tuples[k];
		vector<Real> scaledSnew = data->standardize(_t->s);
		net->predict(scaledSnew, Qs, timeSeries_base, k);
	}

	const Uint thisAction = aInfo.actionToLabel(data->Tmp[agentId]->tuples[ndata-1]->a);
	//sensitivity of value for this action in this state wrt all previous inputs
	for (Uint ii=0; ii<ndata; ii++)
		for (Uint i=0; i<nInputs; i++) {
			vector<Activation*> timeSeries_diff = net->allocateUnrolledActivations(ndata);

			for (Uint k=0; k<ndata; k++) {
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
	if (!out.good()) _die("Unable to open save into file %s\n", fname.c_str());
	for (Uint k=0; k<ndata; k++) {
		for (Uint j=0; j<nInputs; j++)
			out << timeSeries_base[k]->errvals[j] << " ";
		out << "\n";
	}
	out.close();

	net->deallocateUnrolledActivations(&timeSeries_base);
}

void NFQ::Train_BPTT(const Uint seq, const Uint thrID) const
{
	assert(net->allocatedFrozenWeights && bTrain);
	vector<Real> Qs(nOutputs),Qhats(nOutputs),Qtildes(nOutputs),errs(nOutputs);
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Activation*> timeSeries = net->allocateUnrolledActivations(ndata-1);
	Activation* tgtActivation = net->allocateActivation();

	{   //first prediction in sequence without recurrent connections
		const Tuple * const _t = data->Set[seq]->tuples[0];
		const vector<Real> scaledSold = data->standardize(_t->s);
		net->predict(scaledSold, Qhats, timeSeries, 0);
	}

	for (Uint k=0; k<ndata-1; k++)
	{ //state in k=[0:N-2], act&rew in k+1
		Qs = Qhats; //Q(sNew) predicted at previous loop with moving wghts is current Q
		const Tuple * const _t = data->Set[seq]->tuples[k+1]; //this tuple contains a, sNew, reward
		const bool terminal = k+2==ndata && data->Set[seq]->ended;

		if (not terminal)
		{
			const vector<Real> scaledSnew = data->standardize(_t->s);
			net->predict(scaledSnew, Qtildes, timeSeries[k], tgtActivation,
					net->tgt_weights, net->tgt_biases);

			if (k+2==ndata)
				net->predict(scaledSnew, Qhats, timeSeries[k], tgtActivation);
			else  //used for next transition:
			net->predict(scaledSnew, Qhats, timeSeries, k+1);
		}

		// find best action for sNew with moving wghts, evaluate it with tgt wgths:
		// Double Q Learning ( http://arxiv.org/abs/1509.06461 )
		const Uint indBest = maxInd(Qhats);
		const Real target = (terminal) ? _t->r : _t->r + gamma*Qtildes[indBest];
		const Uint action = aInfo.actionToLabel(_t->a);
		const Real err =  (target - Qs[action]);
		for (Uint i=0; i<nOutputs; i++) errs[i] = i==action ? err : 0;

		net->setOutputDeltas(errs, timeSeries[k]);
		dumpStats(Vstats[thrID], Qs[action], err, Qs);
		data->Set[seq]->tuples[k]->SquaredError = err*err;
	}

	if (thrID==0) net->backProp(timeSeries, net->grad);
	else net->backProp(timeSeries, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&timeSeries);
	_dispose_object(tgtActivation);
}

void NFQ::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	assert(net->allocatedFrozenWeights && bTrain);
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Real> Qs(nOutputs),Qhats(nOutputs),Qtildes(nOutputs),errs(nOutputs);

	const Tuple* const _t = data->Set[seq]->tuples[samp+1];
	const Tuple* const t_ = data->Set[seq]->tuples[samp];

	const vector<Real> scaledSold = data->standardize(t_->s);
	Activation* sOldActivation = net->allocateActivation();
	net->predict(scaledSold, Qs, sOldActivation);

	const bool terminal = samp+2==ndata && data->Set[seq]->ended;
	if (not terminal)
	{
		const vector<Real> scaledSnew = data->standardize(_t->s);
		//vector<Real> scaledSnew = data->standardize(_t->s, __NOISE, thrID);
		Activation* sNewActivation = net->allocateActivation();
		net->predict(scaledSnew, Qhats,   sNewActivation);
		net->predict(scaledSnew, Qtildes, sNewActivation,
				net->tgt_weights, net->tgt_biases);
		_dispose_object(sNewActivation);
	}

	// find best action for sNew with moving wghts, evaluate it with tgt wgths:
	// Double Q Learning ( http://arxiv.org/abs/1509.06461 )
	const Uint indBest = maxInd(Qhats);
	const Real target = (terminal) ? _t->r : _t->r + gamma*Qtildes[indBest];
	const Uint action = aInfo.actionToLabel(_t->a);
	const Real err =  (target - Qs[action]);
	for (Uint i=0; i<nOutputs; i++) errs[i] = i==action ? err : 0;

	dumpStats(Vstats[thrID], Qs[action], err, Qs);
	data->Set[seq]->tuples[samp]->SquaredError = err*err;

	if (thrID==0) net->backProp(errs, sOldActivation, net->grad);
	else net->backProp(errs, sOldActivation, net->Vgrad[thrID]);
	_dispose_object(sOldActivation);
}
