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

DACER::DACER(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm,_env,settings,settings.nnOutputs+2), nA(env->aI.maxLabel),
truncation(1000), delta(1), generators(settings.generators)
{
	assert(nOutputs == 1+nA+nA);
	buildNetwork(net, opt, net_outputs, settings);
	assert(nOutputs == net->getnOutputs());
	assert(nInputs == net->getnInputs());
	//This algorithm does not require Recurrent nets, but it does require sampling
	//sequences instead of transitions, so this is a workaround:
	data->bRecurrent = bRecurrent = true;
	test();
}

void DACER::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	vector<Real> output = output_stochastic_policy(agentId,s,a,sOld,aOld,info,r);
	if (output.size() == 0) return;
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

	#if 1
	beta.insert(beta.end(), pol.vals.begin(), pol.vals.end());
	#endif
	data->passData(agentId, info, sOld, a, beta, s, r);
}

void DACER::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	die("DACER requires sampling sequences.\n");
}

void DACER::Train_BPTT(const Uint seq, const Uint thrID) const
{
	//this should go to gamma rather quick:
	const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
	const Real rGamma = annealedGamma();
	assert(net->allocatedFrozenWeights && bTrain);
	const Uint ndata = data->Set[seq]->tuples.size();

	if(thrID==1) profiler->push_start("F");
	vector<Activation*> series_cur = net->allocateUnrolledActivations(ndata-1);
	vector<Activation*> series_hat = net->allocateUnrolledActivations(ndata);

	for (Uint k=0; k<ndata-1; k++) {
		const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains s, a, mu
		const vector<Real> scaledSold = data->standardize(_t->s);
		//const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
		net->seqPredict_inputs(scaledSold, series_cur[k]);
		net->seqPredict_inputs(scaledSold, series_hat[k]);
	}
	net->seqPredict_execute(series_cur, series_cur);
	net->seqPredict_execute(series_cur, series_hat, net->tgt_weights, net->tgt_biases);

	Real Q_RET = 0, Q_OPC = 0;
	//if partial sequence then compute value of last state (=! R_end)
	if(not data->Set[seq]->ended)
	{
		const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
		vector<Real> out_T(nOutputs, 0), S_T = data->standardize(_t->s); //last state
		net->predict(S_T, out_T, series_hat, ndata-1, net->tgt_weights, net->tgt_biases);
		Q_RET = Q_OPC = out_T[0]; //V(s_T) computed with tgt weights
	}
#ifndef NDEBUG
	else assert(data->Set[seq]->tuples[ndata-1]->mu.size() == 0);
#endif

	if(thrID==1) {
		profiler->pop_stop();
		profiler->push_start("L");
	}
	for (int k=static_cast<int>(ndata)-2; k>=0; k--)
	{
		const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains sOld, a
		const Tuple * const t_ = data->Set[seq]->tuples[k+1]; //this contains r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		//get everybody camera ready:
		vector<Real> out_cur = net->getOutputs(series_cur[k]);
		vector<Real> out_hat = net->getOutputs(series_hat[k]);
		const Real V_cur = out_cur[net_indices[0]];
		const Real V_hat = out_hat[net_indices[0]];
		const Discrete_policy pol_cur = prepare_policy(out_cur);
		const Discrete_policy pol_hat = prepare_policy(out_hat);
		//off policy stored action and on-policy sample:
		const Uint act = aInfo.actionToLabel(_t->a); //unbounded action space
		const Uint pol = pol_cur.sample(&generators[thrID]);

		const Real rho_cur = pol_cur.probability(act)/_t->mu[act];
		const Real rho_pol = pol_cur.probability(pol)/_t->mu[pol];
		const Real rho_hat = pol_hat.probability(act)/_t->mu[act];
		//const Real c_cur = std::min((Real)1.,rho_cur);
		const Real c_hat = std::min((Real)1.,rho_hat);
		const Real varCritic = pol_cur.advantageVariance();

		//compute Q using tgt net for pi and C, for consistency of derivatives
		const Real A_cur = pol_cur.computeAdvantage(act);
		const Real A_pol = pol_cur.computeAdvantage(pol);
		const Real A_hat = pol_hat.computeAdvantage(act);
		//compute quantities needed for trunc import sampl with bias correction
		const Real importance = std::min(rho_cur, truncation);
		const Real correction = std::max(0., 1.-truncation/rho_pol);
		const Real A_OPC = Q_OPC - V_hat;
		static const Real L = 0.25, eps = 2.2e-16;
		const Real threshold = A_cur * A_cur / (varCritic+eps);
		const Real smoothing = threshold>L ? L/(threshold+eps) : 2-threshold/L;
		const Real eta = anneal * smoothing * A_cur * A_OPC / (varCritic+eps);

#ifdef ACER_PENALIZER
		const Real cotrolVar = A_pol;
#else
		const Real cotrolVar = 0;
#endif
		const Real gain1 = A_OPC * importance - eta * rho_cur * cotrolVar;
		//const Real gain1 = A_OPC * importance - eta * cotrolVar;
		const Real gain2 = A_pol * correction;
		const vector<Real> gradAcer_1 = pol_cur.policy_grad(act, gain1);
		const vector<Real> gradAcer_2 = pol_cur.policy_grad(pol, gain2);
#ifdef ACER_PENALIZER
		const vector<Real> gradC = pol_cur.control_grad(eta);
		const vector<Real> policy_grad = sum3Grads(gradAcer_1, gradAcer_2, gradC);
#else
		const vector<Real> policy_grad = sum2Grads(gradAcer_1, gradAcer_2);
#endif

		//trust region updating
		const vector<Real> gradDivKL = pol_cur.div_kl_grad(&pol_hat);
		const vector<Real> trust_grad=trust_region_update(policy_grad,gradDivKL,delta);
		const Real Qer = (Q_RET -A_cur -V_cur);
		//const Real Ver = (Q_RET-A_cur-V_cur)*std::min(1.,rho_hat); //unclear usefulness:

		//prepare rolled Q with off policy corrections for next step:
		Q_RET = c_hat* minAbsValue(Q_RET-A_hat-V_hat,Q_RET-A_cur-V_cur)+minAbsValue(V_hat,V_cur);
		//Q_OPC = c_cur*1.*(Q_OPC -A_hat -out_hat[k][0]) +Vs;
		Q_OPC =   0.5* minAbsValue(Q_OPC-A_hat-V_hat,Q_OPC-A_cur-V_cur)+minAbsValue(V_hat,V_cur);

		vector<Real> gradient(nOutputs,0);
		gradient[net_indices[0]]= Qer;
		pol_cur.values_grad(act, Qer, gradient);
		pol_cur.finalize_grad(trust_grad, gradient);

		//bookkeeping:
		dumpStats(Vstats[thrID], A_cur+V_cur, Qer);
		data->Set[seq]->tuples[k]->SquaredError =Qer*Qer;
		vector<Real> _dump = gradient; _dump.push_back(gain1); _dump.push_back(eta);
		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], _dump);

		//write gradient onto output layer:
		clip_gradient(gradient, stdGrad[0]);
		net->setOutputDeltas(gradient, series_cur[k]);
	}

	if(thrID==1) {
		profiler->pop_stop();
		profiler->push_start("B");
	}

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	net->deallocateUnrolledActivations(&series_hat);

	if(thrID==1) profiler->pop_stop();
}
