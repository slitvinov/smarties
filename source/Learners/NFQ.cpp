/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 * TODO:
  - change nOutputs to nA to allow for aux tasks
	- add aux tasks
	- add option to switch between double Q and V-A formulation
 */

#include "NFQ.h"
#include "../Math/Utils.h"

NFQ::NFQ(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm,_env,settings,settings.nnOutputs)
{
	#ifdef NDEBUG
	//if(bRecurrent) die("NFQ recurrent not tested!\n");
	#endif
	buildNetwork(net, opt, vector<Uint>(1,nOutputs), settings);
	printf("NFQ: Built network\n");
	assert(nOutputs == net->getnOutputs());
	assert(nInputs == net->getnInputs());
	policyVecDim = nOutputs;
}

void NFQ::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	vector<Real> beta(policyVecDim,0);
	if(info==2) { data->passData(agentId, info, sOld, a, s, r, beta); return; }
	vector<Real> output = output_value_iteration(agentId,s,a,sOld,aOld,info,r);
	//load computed policy into a
	const Uint indBest = maxInd(output);
	//random action?
	const Real anneal = annealingFactor();
	const Real annealedEps = bTrain ? anneal + (1-anneal)*greedyEps : greedyEps;
	uniform_real_distribution<Real> dis(0.,1.);

	if(dis(*gen) < annealedEps)
		a.set(nOutputs*dis(*gen));
	else
		a.set(indBest);

	for(Uint k=0; k<nOutputs; k++)
		beta[k] = annealedEps/nOutputs + (indBest==k ? (1-annealedEps) : 0);

	data->passData(agentId, info, sOld, a, s, r, beta);
	dumpNetworkInfo(agentId);
}

void NFQ::Train_BPTT(const Uint seq, const Uint thrID) const
{
	const Real rGamma = annealedGamma();
	const Uint ndata = data->Set[seq]->tuples.size();
	const Uint nValues = data->Set[seq]->ended ? ndata-1 :ndata;
	vector<Activation*> actcur = net->allocateUnrolledActivations(nValues);
	vector<Activation*> acthat = net->allocateUnrolledActivations(nValues);

	for (Uint k=0; k<nValues; k++) {
		const vector<Real> inp = data->standardize(data->Set[seq]->tuples[k]->s);
		net->seqPredict_inputs(inp, actcur[k]);
		net->seqPredict_inputs(inp, acthat[k]);
	}
	net->seqPredict_execute(actcur,actcur);
	net->seqPredict_execute(actcur,acthat,net->tgt_weights,net->tgt_biases);

	for (Uint k=0; k<ndata-1; k++) { //state in k=[0:N-2], act&rew in k+1
		const Tuple * const _t = data->Set[seq]->tuples[k+1]; //contains sNew, rew
		const Tuple * const t_ = data->Set[seq]->tuples[k]; //contains sOld, act
		const bool term = k+2==ndata && data->Set[seq]->ended;
		const vector<Real> Qs = net->getOutputs(actcur[k]);
		const vector<Real>Qhats  =term?vector<Real>():net->getOutputs(actcur[k+1]);
		const vector<Real>Qtilde =term?vector<Real>():net->getOutputs(acthat[k+1]);

		// find best action for sNew with moving wghts, evaluate it with tgt wgths:
		// Double Q Learning ( http://arxiv.org/abs/1509.06461 )
		const Real target = (term) ? _t->r : _t->r + rGamma*Qtilde[maxInd(Qhats)];
		const Uint action = aInfo.actionToLabel(t_->a);
		const Real error  = target - Qs[action];
		vector<Real> gradient(nOutputs,0);
		gradient[action] = error;

		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], gradient);
		data->Set[seq]->tuples[k]->SquaredError = error*error;
		clip_gradient(gradient, stdGrad[0], seq, k);
		dumpStats(Vstats[thrID], Qs[action], error);
		net->setOutputDeltas(gradient, actcur[k]);
	}

	if(not data->Set[seq]->ended) { //terminal value does not get corrected
		delete actcur.back();
		actcur.pop_back();
	}
	if (thrID==0) net->backProp(actcur, net->grad);
	else net->backProp(actcur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&actcur);
	net->deallocateUnrolledActivations(&acthat);
}

void NFQ::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	const Real rGamma = annealedGamma();
	vector<Real> Qhats(nOutputs,0), Qtildes(nOutputs,0), gradient(nOutputs,0);
	const Uint ndata = data->Set[seq]->tuples.size(), nMaxBPTT = MAX_UNROLL_BFORE;
	const Uint iRecurr = bRecurrent ? max(nMaxBPTT,samp)-nMaxBPTT : samp;
	const Uint nRecurr = bRecurrent ? min(nMaxBPTT,samp)+1			  : 1;
	const bool terminal = samp+2==ndata && data->Set[seq]->ended;
	vector<Activation*> series_cur = net->allocateUnrolledActivations(nRecurr);
	Activation* tgtAct = terminal ? nullptr : net->allocateActivation();

	for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
		const Tuple * const _t = data->Set[seq]->tuples[k];
		net->seqPredict_inputs(data->standardize(_t->s), series_cur[j]);
	}
	//all are loaded: execute the whole loop:
	net->seqPredict_execute(series_cur, series_cur);
	//extract the only output we actually correct:
	const vector<Real> Qs = net->getOutputs(series_cur.back());
	const Tuple* const t_ = data->Set[seq]->tuples[samp];
	const Tuple* const _t = data->Set[seq]->tuples[samp+1];
	const Uint act = aInfo.actionToLabel(t_->a);

	if (not terminal) {
		const vector<Real> snew = data->standardize(_t->s);
		net->predict(snew, Qhats,   series_cur.back(), tgtAct);
		net->predict(snew, Qtildes, series_cur.back(), tgtAct, net->tgt_weights, net->tgt_biases);
	}

	// find best action for sNew with moving wghts, evaluate it with tgt wgths:
	// Double Q Learning ( http://arxiv.org/abs/1509.06461 )
	const Real value = (terminal) ? _t->r : _t->r +rGamma*Qtildes[maxInd(Qhats)];
	const Real error = value - Qs[act];
	gradient[act] = error;

	statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], gradient);
	data->Set[seq]->tuples[samp]->SquaredError = error*error;
	clip_gradient(gradient, stdGrad[0], seq, samp);
	dumpStats(Vstats[thrID], Qs[act], error);
	net->setOutputDeltas(gradient, series_cur.back());
	//if(thrID==1) {printf("%d %u %u %u %u\n",terminal,samp,ndata,iRecurr,nRecurr);fflush(0);}

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	_dispose_object(tgtAct);
}
