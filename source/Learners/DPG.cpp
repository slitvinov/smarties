/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "../Math/Utils.h"
#include "DPG.h"

DPG::DPG(MPI_Comm comm, Environment*const _env, Settings & _s) :
Learner_utils(comm,_env,_s,_s.nnOutputs), nA(_env->aI.dim),
nS(_env->sI.dimUsed*(1+_s.appendedObs)), cntValGrad(nThreads+1,0),
avgValGrad(nThreads+1,vector<Real>(1,0)), stdValGrad(nThreads+1,vector<Real>(1,0))
{
	#ifdef NDEBUG
	if(bRecurrent) die("DPG with RNN is Not ready!\n");
	#endif
	const vector<Real> out_weight_inits = {_s.outWeightsPrefac};
	buildNetwork(net_value, opt_value, vector<Uint>(1,1), _s, vector<Real>(), vector<Uint>(1,nA));
	buildNetwork(net, opt, vector<Uint>(1,nA), _s, out_weight_inits);
	policyVecDim = 2*nA;
}

void DPG::select(const int agentId, State& s, Action& a,
		State& sOld, Action& aOld, const int info, Real r)
{
	vector<Real> beta(policyVecDim,0);
	if(info==2) { data->passData(agentId, info, sOld, a, s, r, beta); return; }

	vector<Real> output = output_value_iteration(agentId,s,a,sOld,aOld,info,r);
	const Real annealedVar = bTrain ? .2*annealingFactor()+greedyEps : greedyEps;

	if(positive(annealedVar)) {
		std::normal_distribution<Real> dist(0, annealedVar);
		for(Uint i=0; i<nA; i++) {
			beta[i] = output[i];
			beta[i+nA] = 1/annealedVar/annealedVar;
			output[i] += dist(*gen);
		}
	}

	//scale back to action space size:
	a.set(aInfo.getScaled(output));
	data->passData(agentId, info, sOld, a, s, r, beta);
	dumpNetworkInfo(agentId);
}

void DPG::Train_BPTT(const Uint seq, const Uint thrID) const
{
	const Real rGamma = annealedGamma();
	const Uint ndata = data->Set[seq]->tuples.size();
	const Uint ntgts = data->Set[seq]->ended ? ndata-1 : ndata;
	Grads* tmp_grad = new Grads(net_value->getnWeights(),net_value->getnBiases());
	vector<Activation*>valSeries=net_value->allocateUnrolledActivations(ndata-1);
	vector<Activation*>polSeries=net->allocateUnrolledActivations(ndata);
	Activation* tgtAct = net_value->allocateActivation();
	vector<Real> qcurrs(ndata-1), vnexts(ndata-1);

	for (Uint k=0; k<ntgts; k++)
	{ //state in k=[0:N-2], act&rew in k+1, last state (N-1) not used for Q update
		const Tuple*const t_  = data->Set[seq]->tuples[k]; //contains sOld
		vector<Real> s = data->standardize(t_->s);
		vector<Real> pol(nA), val(1), polgrad(nA);
		net->predict(s, pol, polSeries, k); //Compute policy with state as input

		//Advance target network with state and policy
		s.insert(s.end(), pol.begin(), pol.end());
		//Prev step action "a" was performed, not policy. Therefore, next target is
		//computed with recur inputs from value-net computed with cur weights:
		const Activation*const recur = k>0 ? valSeries[k-1] : nullptr;
		net_value->predict(s, val, recur, tgtAct, net_value->tgt_weights, net_value->tgt_biases);
		if(k) vnexts[k-1] = val[0];

		if(k==ndata-1) continue;
		//Advance current network with state and action
		const vector<Real> a = aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
		for(Uint i=0; i<nA; i++) s[nInputs+i] = a[i];
		net_value->predict(s, val, valSeries, k); //Compute value
		qcurrs[k] = val[0];

		//only one-step backprop because policy-net tries to maximize Q given past transitions, so cannot affect previous Q
		net_value->backProp(vector<Real>(1,1), tgtAct, net_value->tgt_weights, net_value->tgt_biases, tmp_grad);

		for(Uint j=0; j<nA; j++) polgrad[j]= tgtAct->errvals[net_value->iInp[nS+j]];
		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], polgrad);
		clip_gradient(polgrad, stdGrad[0], seq, k);
		net->setOutputDeltas(polgrad, polSeries[k]);
	}

	//im done using the term state for the policy, and i want to bptt:
	delete polSeries.back();
	polSeries.pop_back();
	if (thrID==0) net->backProp(polSeries, net->grad);
	else net->backProp(polSeries, net->Vgrad[thrID]);

	for (Uint k=0; k<ndata-1; k++)
	{
		vector<Real> gradient(1);
		const Tuple*const _t  = data->Set[seq]->tuples[k+1];
		const bool terminal = k+2==ndata && data->Set[seq]->ended;
		const Real target = (terminal) ? _t->r : _t->r + rGamma*vnexts[k];
		gradient[0] = target - qcurrs[k];
		data->Set[seq]->tuples[k]->SquaredError = gradient[0]*gradient[0];
		statsGrad(avgValGrad[thrID+1],stdValGrad[thrID+1],cntValGrad[thrID+1],gradient);
		clip_gradient(gradient, stdValGrad[0], seq, k);
		net_value->setOutputDeltas(gradient, valSeries[k]);
		dumpStats(Vstats[thrID], qcurrs[k], gradient[0]);
	}
	if (thrID==0) net_value->backProp(valSeries, net_value->grad);
	else net_value->backProp(valSeries, net_value->Vgrad[thrID]);

	net_value->deallocateUnrolledActivations(&valSeries);
	net->deallocateUnrolledActivations(&polSeries);
	_dispose_object(tgtAct);
	_dispose_object(tmp_grad);
}

void DPG::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	const Real rGamma = annealedGamma();
	const Uint ndata = data->Set[seq]->tuples.size(), nMaxBPTT = MAX_UNROLL_BFORE;
	const Uint iRecurr = bRecurrent ? max(nMaxBPTT,samp)-nMaxBPTT : samp;
	const Uint nRecurr = bRecurrent ? min(nMaxBPTT,samp)+1 : 1;
	const bool terminal = samp+2==ndata && data->Set[seq]->ended;
	vector<Activation*> actPolcur=net->allocateUnrolledActivations(nRecurr);
	vector<Activation*> actValcur=net_value->allocateUnrolledActivations(nRecurr);
	Grads*const tmp = new Grads(net_value->getnWeights(),net_value->getnBiases());
	Activation* tgtVal = net_value->allocateActivation();
	Activation* tgtPol = net->allocateActivation();
	vector<Real> vnext(1), vcurr(1), grad_pol(nA), qcurr(1), grad_val(1,1);
	//number of state inputs to value net, =/= nS in case multiple obs fed
	const Uint NSIN = net_value->getnInputs()-nA;

	for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
		const vector<Real> a = aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
		vector<Real> s = data->standardize(data->Set[seq]->tuples[k]->s);
		net->seqPredict_inputs(s, actPolcur[j]);
		s.insert(s.end(), a.begin(), a.end());
		net_value->seqPredict_inputs(s, actValcur[j]);
		if(k==samp) {
			net->seqPredict_execute(actPolcur,actPolcur);
			net_value->seqPredict_execute(actValcur,actValcur);
			const vector<Real> pol = net->getOutputs(actPolcur.back());
			qcurr = net_value->getOutputs(actValcur.back());
			for(Uint i=0; i<nA; i++) s[NSIN+i] = pol[i];
			net_value->predict(s, vcurr, actValcur[j-1], tgtVal, net_value->tgt_weights, net_value->tgt_biases);
		}
	}
	net_value->backProp(grad_val, tgtVal, net_value->tgt_weights, net_value->tgt_biases, tmp);
	for(Uint j=0;j<nA;j++) grad_pol[j] = tgtVal->errvals[net_value->iInp[NSIN+j]];
	statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad_pol);
	clip_gradient(grad_pol, stdGrad[0], seq, samp);
	net->setOutputDeltas(grad_pol, actPolcur.back());

	const Tuple * const _t = data->Set[seq]->tuples[samp+1]; //contains sNew, rew
	if(!terminal) {
		vector<Real> snew = data->standardize(_t->s), polnext(nA);
		net->predict(snew, polnext, tgtPol, net->tgt_weights, net->tgt_biases);
		snew.insert(snew.end(), polnext.begin(), polnext.end());
		net_value->predict(snew, vnext, tgtVal, net_value->tgt_weights, net_value->tgt_biases);
	}

	const Real target = (terminal) ? _t->r : _t->r + rGamma * vnext[0];
	grad_val[0] = target - qcurr[0];

	data->Set[seq]->tuples[samp]->SquaredError = grad_val[0]*grad_val[0];
	dumpStats(Vstats[thrID], qcurr[0], grad_val[0]);
	statsGrad(avgValGrad[thrID+1], stdValGrad[thrID+1], cntValGrad[thrID+1], grad_val);
	clip_gradient(grad_val, stdValGrad[0], seq, samp);
	net_value->setOutputDeltas(grad_val, actValcur.back());

	if(thrID==0) net_value->backProp(actValcur, net_value->grad);
	else net_value->backProp(actValcur, net_value->Vgrad[thrID]);
	if(thrID==0) net->backProp(actPolcur, net->grad);
	else net->backProp(actPolcur, net->Vgrad[thrID]);

	net->deallocateUnrolledActivations(&actValcur);
	net->deallocateUnrolledActivations(&actPolcur);
	_dispose_object(tgtVal);
	_dispose_object(tgtPol);
	_dispose_object(tmp);
}

void DPG::updateTargetNetwork()
{
	assert(bTrain);
	if (cntUpdateDelay <= 0) { //DQN-style frozen weight
		cntUpdateDelay = tgtUpdateDelay;
		opt_value->moveFrozenWeights(tgtUpdateAlpha);
		opt->moveFrozenWeights(tgtUpdateAlpha);
	}
	if(cntUpdateDelay>0) cntUpdateDelay--;
}

void DPG::stackAndUpdateNNWeights(const Uint nAddedGradients)
{
	assert(nAddedGradients>0 && bTrain);
	opt_value->nepoch ++;
	opt_value->stackGrads(net_value->grad, net_value->Vgrad); //add up gradients across threads
	opt_value->update(net_value->grad, nAddedGradients); //update

	opt->nepoch ++;
	opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads
	opt->update(net->grad, nAddedGradients); //update
}

void DPG::updateNNWeights(const Uint nAddedGradients)
{
	assert(nAddedGradients>0 && bTrain);
	opt_value->nepoch ++;
	opt_value->update(net_value->grad, nAddedGradients);

	opt->nepoch ++;
	opt->update(net->grad, nAddedGradients);
}

void DPG::processGrads()
{
	statsVector(avgGrad, stdGrad, cntGrad);
	statsVector(avgValGrad, stdValGrad, cntValGrad);
	std::ostringstream o1; o1 << "Grads avg (std): ";
	for (Uint i=0;i<avgGrad[0].size();i++)
		o1<<avgGrad[0][i]<<" ("<<stdGrad[0][i]<<") ";
	for (Uint i=0;i<avgValGrad[0].size();i++)
		o1<<avgValGrad[0][i]<<" ("<<stdValGrad[0][i]<<") ";
	cout<<o1.str()<<endl;

	ofstream filestats;
	filestats.open("grads.txt", ios::app);
	filestats<<print(avgGrad[0]).c_str()<<" "<<print(stdGrad[0]).c_str()<<" "
				<<print(avgValGrad[0]).c_str()<<" "<<print(stdValGrad[0]).c_str()<<endl;
	filestats.close();
}
