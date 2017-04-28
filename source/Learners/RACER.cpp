/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "RACER.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cmath>

RACER::RACER(MPI_Comm comm, Environment*const _env, Settings & settings) :
PolicyAlgorithm(comm,_env,settings, 0.1), truncation(5)
{
	#if defined __RELAX
		// I output V(s), P(s), pol(s), prec(s) (and variate)
		#ifdef __VARIATE
			const vector<int> noutputs = {1,nL,nA,nA,1};
			assert(nOutputs == 1+nL+nA+nA+1);
		#else
			const vector<int> noutputs = {1,nL,nA,nA};
			assert(nOutputs == 1+nL+nA+nA);
		#endif
	#elif defined __SAFE
		// I output V(s), P(s), pol(s), mu(s) (and variate)
		#ifdef __VARIATE
			const vector<int> noutputs = {1,nL,nA,nA,1};
			assert(nOutputs == 1+nL+nA+nA+1);
		#else
			const vector<int> noutputs = {1,nL,nA,nA};
			assert(nOutputs == 1+nL+nA+nA);
		#endif
	#else //full formulation
		// I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
		#ifdef __VARIATE
			const vector<int> noutputs = {1,nL,nA,nA,nA,1};
			assert(nOutputs == 1+nL+nA+nA+nA+1);
		#else
			const vector<int> noutputs = {1,nL,nA,nA,nA};
			assert(nOutputs == 1+nL+nA+nA+nA);
		#endif
	#endif

	buildNetwork(noutputs, settings);
	assert(nOutputs == net->getnOutputs());
	assert(nInputs == net->getnInputs());

	checkGradient();
}

void RACER::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	vector<Real> output = basicNetOut(agentId, s, a, sOld, aOld, info, r);
	if (output.size() == 0) return;
	assert(output.size() == nOutputs);
	//variance is pos def: transform linear output layer with softplus

	const vector<Real> mu = extractPolicy(output);
	const vector<Real> prec = extractPrecision(output);
	const vector<Real> var = extractVariance(output);

	const vector<Real> beta = basicNetOut(a, mu, var);
	assert(beta.size() == 2*nA);
	data->passData(agentId, info, sOld, a, beta, s, r);
}

void RACER::Train(const int seq, const int samp, const int thrID) const
{
	die("RACER only works by sampling entire trajectories.\n");
}

void RACER::Train_BPTT(const int seq, const int thrID) const
{
	//this should go to gamma rather quick:
	#ifdef __VARIATE
	const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
	#endif
	const Real rGamma = annealedGamma();

	assert(net->allocatedFrozenWeights && bTrain);
	const int ndata = data->Set[seq]->tuples.size();
	vector<vector<Real>> out_cur(ndata-1, vector<Real>(nOutputs,0));
	vector<vector<Real>> out_hat(ndata-1, vector<Real>(nOutputs,0));
	vector<Activation*> series_cur = net->allocateUnrolledActivations(ndata-1);
	vector<Activation*> series_hat = net->allocateUnrolledActivations(ndata);

	for (int k=0; k<ndata-1; k++)
	{
		const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains s, a, mu
		const vector<Real> scaledSold = data->standardize(_t->s);
		//const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
		net->predict(scaledSold, out_cur[k], series_cur, k);
		net->predict(scaledSold, out_hat[k], series_hat, k, net->tgt_weights, net->tgt_biases);
	}

	Real Q_RET = 0, Q_OPC = 0;
	//if partial sequence then compute value of last state (=! R_end)
	if(not data->Set[seq]->ended)
	{
		const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
		vector<Real> out_T(nOutputs, 0), S_T = data->standardize(_t->s); //last state
		net->predict(S_T, out_T, series_hat, ndata-1, net->tgt_weights, net->tgt_biases);
		Q_RET = out_T[0]; //V(s_T) computed with tgt weights
		//net->predict(S_T, out_T, series_cur.back(), series_hat.back());
		Q_OPC = out_T[0]; //V(s_T) computed with tgt weights
	}
	#ifndef NDEBUG
		else
		{
			const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
			assert(_t->mu.size() == 0);
		}
	#endif

	for (int k=ndata-2; k>=0; k--)
	{
		const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains sOld, a
		const Tuple * const t_ = data->Set[seq]->tuples[k+1]; //this contains r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		//get everybody camera ready:
		//mean of the stochastic policy:
		const vector<Real> polCur = extractPolicy(out_cur[k]);
		const vector<Real> polHat = extractPolicy(out_hat[k]);
		//pos def matrix for quadratic Q:
		const vector<Real> P_Cur = preparePmatrix(out_cur[k]);
		const vector<Real> P_Hat = preparePmatrix(out_hat[k]);

		#ifndef __RELAX
			//location of max of quadratic Q
			const vector<Real> mu_Cur = extractQmean(out_cur[k]);
			const vector<Real> mu_Hat = extractQmean(out_hat[k]);
		#else
			const vector<Real> mu_Cur = polCur;
			const vector<Real> mu_Hat = polHat;
		#endif

		#ifndef __SAFE
			//pass through softplus to make it pos def:
			const vector<Real> preCur = extractPrecision(out_cur[k]);
			const vector<Real> preHat = extractPrecision(out_hat[k]);
			const vector<Real> varCur = extractVariance(out_cur[k]);
			const vector<Real> varHat = extractVariance(out_hat[k]);
		#else
			const vector<Real> preCur = preHat = vector<Real>(nA, precision);
			const vector<Real> varCur = varHat = vector<Real>(nA, variance);
		#endif

		//off policy stored action and on-policy sample:
		const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
		const vector<Real> pol = samplePolicy(polCur, varCur, thrID);

		const Real actProbOnPolicy = evaluateLogProbability(act, polCur, preCur);
		const Real polProbOnPolicy = evaluateLogProbability(pol, polCur, preCur);
		const Real actProbOnTarget = evaluateLogProbability(act, polHat, preHat);
		const Real actProbBehavior = evaluateLogBehavioralPolicy(act, _t->mu);
		const Real polProbBehavior = evaluateLogBehavioralPolicy(pol, _t->mu);

		const Real rho_cur = safeExp(actProbOnPolicy-actProbBehavior);
		const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
		const Real rho_hat = safeExp(actProbOnTarget-actProbBehavior);

		const Real c_cur = std::min((Real)1.,std::pow(rho_cur,1./nA));
		const Real c_hat = std::min((Real)1.,std::pow(rho_hat,1./nA));

		//compute Q using tgt net for pi and C, for consistency of derivatives
		//Q(s,a)                     v a	v policy    v quadratic Q parameters
		const Real A_cur = computeAdvantage(act, polHat, varHat, P_Cur, mu_Cur);
		const Real A_hat = computeAdvantage(act, polHat, varHat, P_Hat, mu_Hat);
		const Real A_pol = computeAdvantage(pol, polCur, varCur, P_Hat, mu_Hat);

		#ifdef __VARIATE
			const Real cov_A_A = out_cur[k][nOutputs-1];
			const vector<Real> smp = samplePolicy(polCur, varCur, thrID);
			const Real varCritic = advantageVariance(polCur, varCur, P_Hat, mu_Hat);
			const Real A_tgt = computeAdvantage(smp, polCur, varCur, P_Hat, mu_Hat);
			const Real err_Cov = A_OPC*A_hat - cov_A_A;
			const Real cotrolVar = A_tgt;
			//const Real cotrolVar = nA+diagTerm(varCur,polCur,mu_Hat)
			//												 -diagTerm(varCur,   pol,mu_Hat);

			const Real eta = anneal*std::min(std::max(-.5, cov_A_A/varCritic), 0.5);
			//const Real eta = 0;
		#else
			const Real eta = 0, cotrolVar = 0, err_Cov = 0;
		#endif

		//compute quantities needed for trunc import sampl with bias correction
		const Real importance = std::min(rho_cur, truncation);
		const Real correction = std::max(0., 1.-truncation/rho_pol);
		const Real A_OPC = Q_OPC - out_hat[k][0];

		//const Real gain1 = A_OPC * importance - eta * rho_cur * cotrolVar;
		const Real gain1 = A_OPC * importance - eta * cotrolVar;
		const Real gain2 = A_pol * correction;

		//derivative wrt to statistics
		const vector<Real> gradAcer_1 = policyGradient(mu_Cur, preCur, act, gain1);
		const vector<Real> gradAcer_2 = policyGradient(mu_Cur, preCur, pol, gain2);

		#ifdef __VARIATE
		const vector<Real> gradC = controlGradient(polCur, varCur, P_Hat, mu_Hat, eta);
		const vector<Real> policy_grad = sum3Grads(gradAcer_1, gradAcer_2, gradC);
		#else
		const vector<Real> policy_grad = sum2Grads(gradAcer_1, gradAcer_2);
		#endif

		//trust region updating
		const vector<Real> gradDivKL = gradDKL(mu_Cur, preCur, mu_Hat, preHat);
		const vector<Real> gradAcer = gradAcerTrpo(policy_grad, gradDivKL);

		const Real Qer = (Q_RET -A_cur -out_cur[k][0]);
		//unclear usefulness:
		const Real Ver = (Q_RET -A_cur -out_cur[k][0])*std::min(1.,rho_hat);
		//prepare rolled Q with off policy corrections for next step:
		Q_RET = c_hat*1.*(Q_RET -A_hat -out_hat[k][0]) +out_hat[k][0];
		//TODO: now Q_OPC ios actually Q_RET, which is better?
		//Q_OPC = c_cur*1.*(Q_OPC -A_hat -out_hat[k][0]) +out_hat[k][0];
		Q_OPC = .5*(Q_OPC -A_hat -out_hat[k][0]) + out_hat[k][0];

		const vector<Real> critic_grad =
		criticGradient(P_Cur, polHat, varHat, out_cur[k], mu_Cur, act);
		const vector<Real> grad =
		finalizeGradient(Qer, Ver, critic_grad, policy_grad, out_cur[k], err_Cov);
		//write gradient onto output layer
		net->setOutputDeltas(grad, series_cur[k]);

		//bookkeeping:
		meanGain1[thrID+1] = 0.9999*meanGain1[thrID+1] + 0.0001*gain1;
		meanGain2[thrID+1] = 0.9999*meanGain2[thrID+1] + 0.0001*eta;
		vector<Real> fake{A_cur, 100};
		dumpStats(Vstats[thrID], A_cur+out_cur[k][0], Qer, fake);
		if(thrID == 1) net->updateRunning(series_cur[k]);
		data->Set[seq]->tuples[k]->SquaredError = Qer*Qer;
	}

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	net->deallocateUnrolledActivations(&series_hat);
}

void RACER::processStats(vector<trainData*> _stats, const Real avgTime)
{
	#ifdef __SAFE
		const Real stdev = 0.1 + annealingFactor();
		variance = stdev*stdev;
		precision = 1./variance;
	#endif

	setVecMean(meanGain1); setVecMean(meanGain2);
	printf("Gain terms of policy grad means: [%f] [%f]\n",
	meanGain1[0], meanGain2[0]);
	Learner::processStats(_stats, avgTime);
}

void RACER::dumpPolicy(const vector<Real> lower, const vector<Real>& upper,
		const vector<int>& nbins)
 {
	//a fail in any of these amounts to a big and fat TODO
	if(nAppended || nA!=1) die("TODO missing features\n");
	assert(lower.size() == upper.size());
	assert(nbins.size() == upper.size());
	assert(nbins.size() == nInputs);
	vector<vector<Real>> bins(nbins.size());
	int nDumpPoints = 1;
	for (int i=0; i<nbins.size(); i++) {
		nDumpPoints *= nbins[i];
		bins[i] = vector<Real>(nbins[i]);
		for (int j=0; j<nbins[i]; j++) {
			const Real l = j/(Real)(nbins[i]-1);
			bins[i][j] = lower[i] + (upper[i]-lower[i])*l;
		}
	}

	FILE * pFile = fopen ("dump.txt", "ab");
	vector<Real> output(nOutputs), dump(nInputs+4);

	for (int i=0; i<nDumpPoints; i++)
	{
		vector<Real> state = pickState(bins, i);
		Activation* act = net->allocateActivation();
		net->predict(data->standardize(state), output, act);
		_dispose_object(act);

		vector<Real> mu = extractPolicy(output);
		int cnt = 0;
		dump[cnt++] = output[0];
		dump[cnt++] = aInfo.getScaled(mu[0], 0);

		#ifndef __SAFE
			vector<Real> var =  extractVariance(output);
			dump[cnt++] = std::sqrt(var[0]);
		#else
			dump[cnt++] = std::sqrt(variance);
		#endif

		#ifndef __RELAX
			vector<Real> mean = extractQmean(output);
			dump[cnt++] = aInfo.getScaled(mean[0], 0);
		#else
			dump[cnt++] = aInfo.getScaled(mean[0], 0);
		#endif

		for (int j=0; j<state.size(); j++) dump[cnt++] = state[j];
		assert(cnt == dump.size());
		fwrite(dump.data(),sizeof(Real),state.size()+4,pFile);
	}
	fclose (pFile);
 }
