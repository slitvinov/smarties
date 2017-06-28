/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "DACER.h"
//#define DUMP_EXTRA

DACER::DACER(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm,_env,settings,settings.nnOutputs), nA(env->aI.maxLabel),
truncation(10), delta(0.1), generators(settings.generators)
{
	assert(nOutputs == 1+nA+nA);
	buildNetwork(net, opt, net_outputs, settings);
	assert(nOutputs == net->getnOutputs());
	assert(nInputs == net->getnInputs());
	#ifdef DUMP_EXTRA
		policyVecDim = 2*nA;
	#else
		policyVecDim =   nA;
	#endif
	test();
}

void DACER::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	if (info == 2) { //no need for action, just pass terminal s & r
		data->passData(agentId, info, sOld, a, s, r, vector<Real>(policyVecDim,0));
		return;
	}
	vector<Real> output = output_stochastic_policy(agentId,s,a,sOld,aOld,info,r);
	assert(output.size() == nOutputs);

	const Discrete_policy pol = prepare_policy(output);
	vector<Real> beta = pol.getProbs();
	assert(beta.size()==nA);

	const Real anneal = annealingFactor();
	const Real annealedEps = bTrain ? anneal + (1-anneal)*greedyEps : greedyEps;
	const Real addedVar = annealedEps/nA, trunc = (1-annealedEps);

	if(bTrain && positive(annealedEps))
		for(Uint i=0; i<nA; i++) beta[i] = trunc*beta[i] + addedVar;

	std::discrete_distribution<Uint> dist(beta.begin(),beta.end());
	const Uint iAct = (positive(annealedEps)||bTrain) ? dist(*gen) : maxInd(beta);
	assert(iAct<nA);
	a.set(iAct);

	#ifdef DUMP_EXTRA
	beta.insert(beta.end(), pol.vals.begin(), pol.vals.end());
	#endif
	data->passData(agentId, info, sOld, a, s, r, beta);
	dumpNetworkInfo(agentId);
}

void DACER::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	//this should go to gamma rather quick:
	const Real rGamma = annealedGamma();
	const Uint ndata = data->Set[seq]->tuples.size();
	assert(samp<ndata-1);
	const bool bEnd = data->Set[seq]->ended; //does it have a terminal reward?
	const Uint nMaxTargets = MAX_UNROLL_AFTER+1, nMaxBPTT = MAX_UNROLL_BFORE;
	//for off policy correction we need reward and action, therefore not last one:
	const Uint nSUnroll = min(ndata-1-samp, nMaxTargets);
	//if we do not have a terminal reward, then we compute value of last state:
	const Uint nSValues = min(bEnd? ndata-1-samp : ndata-samp, nMaxTargets);

	const Uint nRecurr = bRecurrent ? min(nMaxBPTT,samp)+1        : 1;
	const Uint iRecurr = bRecurrent ? max(nMaxBPTT,samp)-nMaxBPTT : samp;
	//printf("%d %u %u %u %u\n",bEnd,samp,ndata,nSUnroll,nSValues); fflush(0);
	if(thrID==1) profiler->stop_start("FWD");

	vector<vector<Real>> out_cur(1, vector<Real>(nOutputs,0));
	vector<vector<Real>> out_hat(nSValues, vector<Real>(nOutputs,0));
	vector<Activation*> series_cur = net->allocateUnrolledActivations(nRecurr);
	vector<Activation*> series_hat = net->allocateUnrolledActivations(nSValues);

	for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
		const vector<Real> inp = data->standardize(data->Set[seq]->tuples[k]->s);
		net->seqPredict_inputs(inp, series_cur[j]);
		if(k==samp) { //all are loaded: execute the whole loop:
			assert(j==nRecurr-1);
			net->seqPredict_execute(series_cur, series_cur);
			//extract the only output we actually correct:
			net->seqPredict_output(out_cur[0], series_cur[j]); //the humanity!
			//predict samp with target weight using curr recurrent inputs as estimate:
			const Activation*const recur = j ? series_cur[j-1] : nullptr;
			net->predict(inp, out_hat[0], recur, series_hat[0], net->tgt_weights, net->tgt_biases);
		}
	}

	for (Uint k=1; k<nSValues; k++) {
		const vector<Real>inp= data->standardize(data->Set[seq]->tuples[k+samp]->s);
		net->predict(inp,out_hat[k],series_hat,k,net->tgt_weights,net->tgt_biases);
	}

	if(thrID==1)  profiler->stop_start("ADV");

	Real Q_RET = 0, Q_OPC = 0;
	if(nSValues != nSUnroll) { //partial sequence: compute value of term state
		assert(nSValues==nSUnroll+1 && !bEnd);
		Q_RET=Q_OPC= out_hat[nSValues-1][net_indices[0]]; //V(s_T) with tgt weights
	}

	for (int k=static_cast<int>(nSUnroll)-1; k>0; k--) //propagate Q to k=0
		offPolCorrUpdate(seq, k+samp, Q_RET, Q_OPC, out_hat[k], rGamma);

	if(thrID==1)  profiler->stop_start("CMP");

	vector<Real> grad = compute<0>(seq, samp, Q_RET, Q_OPC, out_cur[0], out_hat[0], rGamma, thrID);

	//#ifdef FEAT_CONTROL
	//const vector<Real> act=aInfo.getInvScaled(data->Set[seq]->tuples[samp]->a);
	//const Activation*const recur = nSValues>1 ? series_hat[1] : nullptr;
	//task->Train(series_cur.back(), recur, act, seq, samp, rGamma, grad);
	//#endif

	//write gradient onto output layer:
	statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
	clip_gradient(grad, stdGrad[0], seq, samp);
	net->setOutputDeltas(grad, series_cur.back());

	if(thrID==1)  profiler->stop_start("BCK");

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	net->deallocateUnrolledActivations(&series_hat);

	if(thrID==1)  profiler->stop_start("TSK");
}

void DACER::Train_BPTT(const Uint seq, const Uint thrID) const
{
	//this should go to gamma rather quick:
	const Real rGamma = annealedGamma();
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Activation*> series_cur = net->allocateUnrolledActivations(ndata-1);
	vector<Activation*> series_hat = net->allocateUnrolledActivations(ndata-1);

	for (Uint k=0; k<ndata-1; k++) {
		const Tuple * const _t = data->Set[seq]->tuples[k]; // s, a, mu
		const vector<Real> scaledSold = data->standardize(_t->s);
		//const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
		net->seqPredict_inputs(scaledSold, series_cur[k]);
		net->seqPredict_inputs(scaledSold, series_hat[k]);
	}
	net->seqPredict_execute(series_cur,series_cur);
	net->seqPredict_execute(series_cur,series_hat,net->tgt_weights,net->tgt_biases);

	Real Q_RET = 0, Q_OPC = 0;
	//if partial sequence then compute value of last state (!= R_end)
	if(not data->Set[seq]->ended) {
		series_hat.push_back(net->allocateActivation());
		const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
		vector<Real> out_T(nOutputs, 0), S_T = data->standardize(_t->s);//last state
		net->predict(S_T,out_T,series_hat,ndata-1,net->tgt_weights,net->tgt_biases);
		Q_OPC = Q_RET = out_T[net_indices[0]]; //V(s_T) computed with tgt weights
	}

	for (int k=static_cast<int>(ndata)-2; k>=0; k--)
	{
		vector<Real> out_cur = net->getOutputs(series_cur[k]);
		vector<Real> out_hat = net->getOutputs(series_hat[k]);
		vector<Real> grad = compute<1>(seq, k, Q_RET, Q_OPC, out_cur, out_hat, rGamma, thrID);
		//#ifdef FEAT_CONTROL
		//const vector<Real> act=aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
		//task->Train(series_cur[k], series_hat[k+1], act, seq, k, rGamma, grad);
		//#endif

		//write gradient onto output layer:
		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
		clip_gradient(grad, stdGrad[0], seq, k);
		net->setOutputDeltas(grad, series_cur[k]);
	}

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	net->deallocateUnrolledActivations(&series_hat);
}
