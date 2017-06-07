/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "NAF.h"

NAF::NAF(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm,_env,settings,settings.nnOutputs),
nA(_env->aI.dim), nL(compute_nL(_env->aI.dim))
{
	#ifdef NDEBUG
	//if(bRecurrent) die("NAF recurrent not tested!\n");
	#endif
	vector<Real> out_weight_inits = {-1, -1, settings.outWeightsPrefac};

	#ifdef FEAT_CONTROL
	const Uint task_out0 = ContinuousSignControl::addRequestedLayers(nA,
		 env->sI.dimUsed, net_indices, net_outputs, out_weight_inits);
	#endif

	buildNetwork(net, opt, net_outputs, settings, out_weight_inits);
	printf("NAF: Built network with outputs: %s %s\n",
		print(net_indices).c_str(), print(net_outputs).c_str());
	assert(nOutputs == net->getnOutputs());
	assert(nInputs == net->getnInputs());
	#ifdef FEAT_CONTROL
	task = new ContinuousSignControl(task_out0, nA, env->sI.dimUsed, net, data);
	#endif
	test();
}

void NAF::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	vector<Real> output = output_value_iteration(agentId,s,a,sOld,aOld,info,r);
	if(!output.size()) {
		assert(info==2);
		return;
	}
	const Quadratic_advantage advantage = prepare_advantage(output);

	//load computed policy into a
	vector<Real> policy = advantage.getMean();
	const Real anneal = annealingFactor();
	const Real annealedVar = bTrain ? anneal + greedyEps : greedyEps;

	if(positive(annealedVar)) {
		std::normal_distribution<Real> dist(0, annealedVar);
		for(Uint i=0; i<nA; i++) policy[i] += dist(*gen);
	}

	//scale back to action space size:
	a.set(aInfo.getScaled(policy));
	dumpNetworkInfo(agentId);
}

void NAF::Train_BPTT(const Uint seq, const Uint thrID) const
{
	const Real rGamma = annealedGamma();
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Real> target(nOutputs), output(nOutputs), gradient(nOutputs);
	vector<Activation*> timeSeries = net->allocateUnrolledActivations(ndata-1);
	Activation* tgtActivation = net->allocateActivation();

	for (Uint k=0; k<ndata-1; k++)
	{ //state in k=[0:N-2], act&rew in k+1, last state (N-1) not used for Q update
		//this tuple contains a, sNew, reward
		const Tuple * const _t    = data->Set[seq]->tuples[k+1];
		//this tuple contains sOld
		const Tuple * const _tOld = data->Set[seq]->tuples[k];
		const vector<Real> sold = data->standardize(_tOld->s);
		const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
		net->predict(sold, output, timeSeries, k);
		Quadratic_advantage adv_sold = prepare_advantage(output);

		const bool terminal = k+2==ndata && data->Set[seq]->ended;
		if (not terminal) {
			//vector<Real> scaledSnew = data->standardize(_t->s, __NOISE, thrID);
			const vector<Real> snew = data->standardize(_t->s);
			net->predict(snew, target, timeSeries[k], tgtActivation,
					net->tgt_weights, net->tgt_biases);
		}

		const Real Vsold = output[net_indices[0]];
		const Real Qsold = Vsold + adv_sold.computeAdvantage(act);
		const Real value = (terminal) ? _t->r : _t->r + rGamma*target[net_indices[0]];
		const Real error = value - Qsold;
		gradient[net_indices[0]] = error;
		adv_sold.grad(act, error, gradient);
		#ifdef FEAT_CONTROL
		task->Train(timeSeries[k],tgtActivation,act,seq,k,rGamma,gradient);
		#endif

		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], gradient);
		clip_gradient(gradient, stdGrad[0]);
		dumpStats(Vstats[thrID], Qsold, error);
		data->Set[seq]->tuples[k]->SquaredError = error*error;
		net->setOutputDeltas(gradient, timeSeries[k]);
	}

	if (thrID==0) net->backProp(timeSeries, net->grad);
	else net->backProp(timeSeries, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&timeSeries);
	_dispose_object(tgtActivation);
}

void NAF::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	const Real rGamma = annealedGamma();
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Real> target(nOutputs), output(nOutputs), gradient(nOutputs);
	Activation* sOldActivation = net->allocateActivation();
	Activation* sNewActivation = net->allocateActivation();
	//this tuple contains sOld:
	const Tuple* const _tOld = data->Set[seq]->tuples[samp];
	//this tuple contains a, sNew, reward:
	const Tuple* const _t = data->Set[seq]->tuples[samp+1];

	const vector<Real> sold = data->standardize(_tOld->s);
	const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
	net->predict(sold, output, sOldActivation); //sOld in previous tuple
	Quadratic_advantage adv_sold = prepare_advantage(output);

	const bool terminal = samp+2==ndata && data->Set[seq]->ended;
	if (not terminal) {
		//vector<Real> scaledSnew = data->standardize(_t->s, __NOISE, thrID);
		vector<Real> snew = data->standardize(_t->s);
		net->predict(snew, target, sNewActivation,net->tgt_weights,net->tgt_biases);
	}

	const Real Vsold = output[net_indices[0]];
	const Real Vsnew = target[net_indices[0]];
	const Real Qsold = Vsold + adv_sold.computeAdvantage(act);
	const Real value = (terminal) ? _t->r : _t->r + rGamma*Vsnew;
	const Real error = value - Qsold;
	gradient[net_indices[0]] = error;
	adv_sold.grad(act, error, gradient);
	#ifdef FEAT_CONTROL
	task->Train(sOldActivation,sNewActivation,act,seq,samp,rGamma,gradient);
	#endif

	statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], gradient);
	clip_gradient(gradient, stdGrad[0]);
	dumpStats(Vstats[thrID], Qsold, error);
	data->Set[seq]->tuples[samp]->SquaredError = error*error;

	if (thrID==0) net->backProp(gradient, sOldActivation, net->grad);
	else net->backProp(gradient, sOldActivation, net->Vgrad[thrID]);
	_dispose_object(sNewActivation);
	_dispose_object(sOldActivation);
}
