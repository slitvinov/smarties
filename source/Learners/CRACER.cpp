/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */


#include "../StateAction.h"
#include "CRACER.h"

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <assert.h>
#include <algorithm>
#include <cmath>



CRACER::CRACER(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner(comm,_env,settings), nA(_env->aI.dim),
nL((_env->aI.dim*_env->aI.dim+_env->aI.dim)/2),
delta(0.1), truncation(5), generators(settings.generators)
{
	printf("Running (R)ACER! Fancy banner here\n");
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
		const int Nnets = 6;
		const int outputs[Nnets] = {1,nL,nA,nA,nA,1};
		const int nsplit = lsize.size()>3 ? 2 : 1;
		for (int i=0; i<lsize.size()-nsplit; i++)
			net->addLayer(lsize[i], lType);

		const int firstSplit = lsize.size()-nsplit;
		const vector<int> lastJointLayer(1,net->getLastLayerID());

		for (int i=0; i<Nnets; i++)
		{
			net->addLayer(lsize[firstSplit], lType, lastJointLayer);

			for (int j=firstSplit+1; j<lsize.size(); j++)
				net->addLayer(lsize[j], lType);

			net->addOutput(outputs[i], "Normal");
		}
	}
	net->build();
	assert(1+nL+3*nA+1 == net->getnOutputs());
	assert(nInputs == net->getnInputs());

	opt = new AdamOptimizer(net, profiler, settings);
	data->bRecurrent = bRecurrent = true;

#if 1//ndef NDEBUG
	vector<Real> out_0(nOutputs, 0.1);
	for(int i = 0; i<nOutputs; i++) {
		uniform_real_distribution<Real> dis(-5,5);
		out_0[i] = dis(*gen);
	}
	vector<Real> act(nA,0.25);
	for(int i = 0; i<nA; i++) {
		uniform_real_distribution<Real> dis(-10,10);
		act[i] = dis(*gen);
	}
	prepareVariance(out_0);

	const vector<Real> polGrad = policyGradient(out_0, act, 1.0);
	for(int i = 0; i<2*nA; i++) {
		vector<Real> grad_1(nOutputs), grad_2(nOutputs);
		vector<Real> out_1 = out_0;
		vector<Real> out_2 = out_0;
		out_1[1+nL+i] -= 0.0001;
		out_2[1+nL+i] += 0.0001;
		const Real p1 = evaluateLogProbability(act, out_1);
		const Real p2 = evaluateLogProbability(act, out_2);

		const Real gradi = polGrad[i];
		const Real diffi = (p2-p1)/0.0002;
		printf("LogPol Gradient %d: finite differences %g analytic %g \n", i, diffi, gradi);
	}
	/*
	vector<Real> grad_0 = computeGradient(1., 0., out_0, out_0, act, polGrad);
	for(int i = 0; i<1+nL; i++) {
		vector<Real> out_1 = out_0;
		vector<Real> out_2 = out_0;
		out_1[i] -= 0.0001;
		out_2[i] += 0.0001;
		const Real Q_1 = computeQ(act, out_0, out_1);
		const Real Q_2 = computeQ(act, out_0, out_2);
		const Real gradi = grad_0[i];
		const Real diffi = (Q_2-Q_1)/0.0002;
		printf("Value Gradient %d: finite differences %g analytic %g \n", i, diffi, gradi);
	}
	for(int i = 0; i<nA; i++) {
		vector<Real> out_1 = out_0;
		vector<Real> out_2 = out_0;
		out_1[1+nL+2*nA+i] -= 0.0001;
		out_2[1+nL+2*nA+i] += 0.0001;
		const Real Q_1 = computeQ(act, out_0, out_1);
		const Real Q_2 = computeQ(act, out_0, out_2);
		const Real gradi = grad_0[1+nL+2*nA+i];
		const Real diffi = (Q_2-Q_1)/0.0002;
		printf("Value Gradient %d: finite differences %g analytic %g \n", i, diffi, gradi);
	}
	*/
#endif

}

void CRACER::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	if (info == 2) { //no need for action, just pass terminal s & r
		data->passData(agentId, info, sOld, a, vector<Real>(), s, r);
		return;
	}

	Activation* currActivation = net->allocateActivation();

	vector<Real> output(nOutputs);
	vector<Real> input = s.copy_observed();
	//if required, chain together nAppended obs to compose state
	if (nAppended>0) {
		const int sApp = nAppended*sInfo.dimUsed;
		if(info==1)
			input.insert(input.end(),sApp, 0);
		else {
			assert(data->Tmp[agentId]->tuples.size()!=0);
			const Tuple * const last = data->Tmp[agentId]->tuples.back();
			input.insert(input.end(),last->s.begin(),last->s.begin()+sApp);
			assert(last->s.size()==input.size());
		}
	}

	if(info==1) {
		net->predict(data->standardize(input), output, currActivation);
	} else { //then if i'm using RNN i need to load recurrent connections (else no effect)
		Activation* prevActivation = net->allocateActivation();
		net->loadMemory(net->mem[agentId], prevActivation);
		net->predict(data->standardize(input), output, prevActivation, currActivation);
		_dispose_object(prevActivation);
	}

	//save network transition
	net->loadMemory(net->mem[agentId], currActivation);
	_dispose_object(currActivation);
	//variance is pos def: transform linear output layer with softplus
	prepareVariance(output);


	const Real eps = annealingFactor();
	if(bTrain && eps) {
		for(int i=0; i<nA; i++) {
			const Real varscale = aInfo.addedVariance(i);
			const Real policy_var = 1./std::sqrt(output[1+nL+nA+i]); //output: 1/S^2
			Real anneal_var = varscale*greedyEps + policy_var;
			//				anneal_var = anneal_var>varscale ? varscale : anneal_var;
			const Real annealed_mean = output[1+nL+i];
			//const Real annealed_mean = output[1+nL+i];
			std::normal_distribution<Real> dist_cur(annealed_mean, anneal_var);
			output[1+nL+i] = annealed_mean; //to save correct mu
			output[1+nL+nA+i] = 1./std::pow(anneal_var, 2); //to save correct mu
			a.vals[i] = dist_cur(*gen);
		}
	}
	else if (greedyEps || bTrain) { //still want to sample policy.
		for(int i=0; i<nA; i++) {
			const Real policy_var = 1./std::sqrt(output[1+nL+nA+i]); //output: 1/S^2
			std::normal_distribution<Real> dist_cur(output[1+nL+i], policy_var);
			a.vals[i] = dist_cur(*gen);
		}
	}
	else {//load computed policy into a
		const vector<Real> pi(&output[1+nL], &output[1+nL]+nA);
		a.set(pi);
	}

	const vector<Real> mu(&output[1+nL], &output[1+nL]+2*nA);
	finalizePolicy(a); //if bounded action space: scale
	data->passData(agentId, info, sOld, a, mu, s, r);

	/*
			#ifdef _dumpNet_
		   if (!bTrain)
				dumpNetworkInfo(agentId);
			#endif
	 */
}

/*
void RACER::dumpNetworkInfo(const int agentId)
{
	net->dump(agentId);
	vector<Real> output(nOutputs);

	const int ndata = data->Tmp[agentId]->tuples.size(); //last one already placed
	if (ndata == 0) return;

	vector<Activation*> timeSeries_base = net->allocateUnrolledActivations(ndata);
	net->clearErrors(timeSeries_base);

	for (int k=0; k<ndata; k++) {
		const Tuple * const _t = data->Tmp[agentId]->tuples[k];
		vector<Real> scaledSnew = data->standardize(_t->s);
		net->predict(scaledSnew, output, timeSeries_base, k);
	}

	//sensitivity of value for this action in this state wrt all previous inputs
	for (int ii=0; ii<ndata; ii++)
	for (int i=0; i<nInputs; i++) {
		vector<Activation*> series =net->allocateUnrolledActivations(ndata);

		for (int k=0; k<ndata; k++) {
			const Tuple* const _t = data->Tmp[agentId]->tuples[k];
			vector<Real> scaledSnew = data->standardize(_t->s);
			if (k==ii) scaledSnew[i] = 0;
			net->predict(scaledSnew, output, series, k);
		}
		vector<Real> oDiff = net->getOutputs(series.back());
		vector<Real> oBase = net->getOutputs(timeSeries_base.back());
		const Tuple* const _t = data->Tmp[agentId]->tuples[ii];
		vector<Real> sSnew = data->standardize(_t->s);

		//assert(nA==1); //TODO ask Sid for muldi-dim actions?
		double dAct = 0;
		for (int j=0; j<nA; j++)
			dAct+=pow(oDiff[1+nL+j]-oBase[1+nL+j],2);

		timeSeries_base[ii]->errvals[i]=sqrt(dAct)/sSnew[i];
		net->deallocateUnrolledActivations(&series);
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
 */

void CRACER::Train(const int seq, const int samp, const int thrID) const
{
	die("RACER only works by sampling entire trajectories.\n");
}

void CRACER::dumpPolicy(const vector<Real> lower, const vector<Real>& upper,
		const vector<int>& nbins)
{
	//a fail in any of these amounts to a big and fat TODO
	if(nAppended || nA!=1)
		die("TODO missing features\n");
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
	vector<Real> Vs(nDumpPoints), Pi(nDumpPoints), Co(nDumpPoints);
	vector<Real> Mu(nDumpPoints), output(nOutputs);
	for (int i=0; i<nDumpPoints; i++) {
		vector<Real> state = pickState(bins, i);
		Activation* act = net->allocateActivation();
		net->predict(data->standardize(state), output, act);
		_dispose_object(act);
		prepareVariance(output);
		Vs[i] = output[0];
		Pi[i] = aInfo.getScaled(output[1+nL], 0);
		Co[i] = 1./std::sqrt(output[1+nL+nA]);
		Mu[i] = aInfo.getScaled(output[1+nL+2*nA], 0);
		vector<Real> dump(state.size()+4);
		dump[0] = Vs[i]; dump[1] = Co[i]; dump[2] = Pi[i]; dump[3] = Mu[i];
		for (int j=0; j<state.size(); j++) dump[j+4] = state[j];
		fwrite(dump.data(),sizeof(Real),dump.size(),pFile);
	}
	fclose (pFile);
}

void CRACER::Train_BPTT(const int seq, const int thrID) const
{
	//this should go to gamma rather quick:
	const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
	const Real rGamma = anneal*gamma;
	//const Real rGamma=std::min(1.,Real(opt->nepoch)/epsAnneal)*gamma;
	//const Real rGamma = gamma;
	assert(net->allocatedFrozenWeights && bTrain);
	const int ndata = data->Set[seq]->tuples.size();
	vector<vector<Real>> out_cur(ndata-1, vector<Real>(1+nL+nA*3+1,0));
	vector<vector<Real>> out_hat(ndata-1, vector<Real>(1+nL+nA*3+1,0));
	vector<Real> rho_cur(ndata-1), rho_pol(ndata-1);
	vector<Real> rho_hat(ndata-1), c_hat(ndata-1);
	vector<Real> c_cur(ndata-1), cov_A_A(ndata-1);
	vector<vector<Real>> act(ndata-1, vector<Real>(nA,0));
	vector<vector<Real>> pol(ndata-1, vector<Real>(nA,0));
	vector<Activation*> series_cur = net->allocateUnrolledActivations(ndata-1);
	vector<Activation*> series_hat = net->allocateUnrolledActivations(ndata);

	for (int k=0; k<ndata-1; k++)
	{
		const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains s, a, mu
		const vector<Real> scaledSold = data->standardize(_t->s);
		//const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
		net->predict(scaledSold, out_cur[k], series_cur, k);
		net->predict(scaledSold, out_hat[k], series_hat, k, net->tgt_weights, net->tgt_biases);
		prepareVariance(out_cur[k]); //pass through softplus to make it pos def
		prepareVariance(out_hat[k]); //grad correction in computeGradient

		act[k] = aInfo.getInvScaled(_t->a); //needs to be unbounded action space
		for(int i=0; i<nA; i++) { //sample current policy
			const Real pol_var = 1./std::sqrt(out_cur[k][1+nL+nA+i]);
			std::normal_distribution<Real> dist_cur(out_cur[k][1+nL+i], pol_var);
			pol[k][i] = dist_cur(generators[thrID]);
		}

		const Real actProbOnPolicy = evaluateLogProbability(act[k], out_cur[k]);
		const Real polProbOnPolicy = evaluateLogProbability(pol[k], out_cur[k]);
		const Real actProbOnTarget = evaluateLogProbability(act[k], out_hat[k]);
		const Real actProbBehavior = evaluateLogBehavioralPolicy(act[k], _t->mu);
		const Real polProbBehavior = evaluateLogBehavioralPolicy(pol[k], _t->mu);

		rho_cur[k] = std::exp(std::min(9.,std::max(-32.,actProbOnPolicy-actProbBehavior)));
		rho_pol[k] = std::exp(std::min(9.,std::max(-32.,polProbOnPolicy-polProbBehavior)));
		rho_hat[k] = std::exp(std::min(9.,std::max(-32.,actProbOnTarget-actProbBehavior)));
		c_cur[k] = std::min((Real)1.,std::pow(rho_cur[k],1./nA));
		c_hat[k] = std::min((Real)1.,std::pow(rho_hat[k],1./nA));
		cov_A_A[k] = out_cur[k].back();
	}

	Real Q_RET = 0, Q_OPC = 0;
	//if partial sequence then compute value of last state (=! R_end)
	if(not data->Set[seq]->ended)
	{
		const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
		vector<Real> S_T = data->standardize(_t->s); //last state
		vector<Real> out_T(1+nL+nA*3+1, 0);
		net->predict(S_T, out_T, series_hat, ndata-1, net->tgt_weights, net->tgt_biases);
		Q_RET = out_T[0]; //V(s_T) computed with tgt weights
		//net->predict(S_T, out_T, series_cur.back(), series_hat.back());
		Q_OPC = out_T[0]; //V(s_T) computed with tgt weights
	}
#ifndef NDEBUG
	else {
		const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
		assert(_t->mu.size() == 0);
	}
#endif

	for (int k=ndata-2; k>=0; k--)
	{
		//const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains sOld
		const Tuple * const t_ = data->Set[seq]->tuples[k+1]; //this contains a, r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		//get everybody camera ready:
		const vector<Real> polCur(&(out_cur[k][1+nL]),     &(out_cur[k][1+nL])     +nA);
		const vector<Real> polHat(&(out_hat[k][1+nL]),     &(out_hat[k][1+nL])     +nA);
		const vector<Real> preCur(&(out_cur[k][1+nL+nA]),  &(out_cur[k][1+nL+nA])  +nA);
		const vector<Real> preHat(&(out_hat[k][1+nL+nA]),  &(out_hat[k][1+nL+nA])  +nA);
		const vector<Real> mu_Cur(&(out_cur[k][1+nL+2*nA]),&(out_cur[k][1+nL+2*nA])+nA);
		const vector<Real> mu_Hat(&(out_hat[k][1+nL+2*nA]),&(out_hat[k][1+nL+2*nA])+nA);
		const vector<Real> varCur = computeVariance(preCur);
		const vector<Real> varHat = computeVariance(preHat);
		const vector<Real> P_Cur = preparePmatrix(out_cur[k]);
		const vector<Real> P_Hat = preparePmatrix(out_hat[k]);
		//compute Q using tgt net for pi and C, for consistency of derivatives
		//Q(s,a)                     v a	v policy    v quadratic Q parameters
		const Real A_cur = computeAdvantage(act[k], polCur, varCur, P_Cur, mu_Cur);
		//const Real A_tgt = computeAdvantage(act[k], polHat, varHat, P_Cur, mu_Cur);
		const Real A_hat = computeAdvantage(act[k], polHat, varHat, P_Hat, mu_Hat);
		const Real A_pol = computeAdvantage(pol[k], polHat, varHat, P_Hat, mu_Hat);
		const Real varCritic = advantageVariance(polCur, varCur, P_Cur, mu_Cur);

		//compute quantities needed for trunc import sampl with bias correction
		const Real importance = std::min(rho_cur[k], truncation);
		const Real correction = std::max(0., 1.-truncation/rho_pol[k]);
		const Real A_OPC = Q_OPC - out_hat[k][0];
		const Real err_Cov = A_OPC*A_cur - cov_A_A[k];

		//const Real gain1 = A_OPC * importance - eta * rho_cur[k] * A_cur;
		//const Real gain2 = A_pol * correction;
		const Real eta = anneal*std::min(std::max(-1., cov_A_A[k]/varCritic), 1.);

		#ifdef __A_VARIATE
		const Real cotrolVar = A_cur;
		#else
		//other ppossible control variate with zero exp. value under policy
		const Real cotrolVar = nA+diagTerm(varCur,polCur,mu_Cur)-diagTerm(varCur,act[k],mu_Cur);
		#endif

		const Real gain1 = rho_cur[k] * (A_OPC - eta * cotrolVar);
		const Real gain2 = 0; //no trunctation and bias reduction
		meanGain1[thrID+1] = 0.9999*meanGain1[thrID+1] + 0.0001*gain1;
		meanGain2[thrID+1] = 0.9999*meanGain2[thrID+1] + 0.0001*eta;
		//derivative wrt to statistics
		const vector<Real> gradAcer_1 = policyGradient(out_cur[k], act[k], gain1);
		const vector<Real> gradAcer_2 = policyGradient(out_cur[k], pol[k], gain2);
		const vector<Real> gradC = controlGradient(polCur, varCur, P_Cur, mu_Cur, eta);
		const vector<Real> policy_grad = sum3Grads(gradAcer_1, gradAcer_2, gradC);

		//trust region updating
		const vector<Real> gradDivKL = gradDKL(out_cur[k], out_hat[k]);
		const vector<Real> gradAcer = gradAcerTrpo(policy_grad, gradDivKL);

		const Real Qer = (Q_RET -A_cur -out_cur[k][0]);
		//unclear usefulness:
		const Real Ver = (Q_RET -A_cur -out_cur[k][0])*std::min(1.,rho_hat[k]);
		//prepare rolled Q with off policy corrections for next step:
		Q_RET = c_hat[k]*1.*(Q_RET -A_hat -out_hat[k][0]) +out_hat[k][0];
		//TODO: now Q_OPC ios actually Q_RET, which is better?
		Q_OPC = c_cur[k]*1.*(Q_OPC -A_hat -out_hat[k][0]) +out_hat[k][0];
		//Q_OPC = .5*(Q_OPC -A_hat -out_hat[k][0]) + out_hat[k][0];
		
		const vector<Real> critic_grad =
		criticGradient(P_Cur, polCur, varCur, out_cur[k], mu_Cur, act[k]);
		const vector<Real> grad =
		finalizeGradient(Qer, Ver, critic_grad, policy_grad, out_cur[k], err_Cov);
		//write gradient onto output layer
		net->setOutputDeltas(grad, series_cur[k]);
		
		//#ifndef NDEBUG
		//printf("Applying gradient %s\n",printVec(grad).c_str());
		//fflush(0);
		//#endif
		
		//bookkeeping:
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

void CRACER::processStats(vector<trainData*> _stats, const Real avgTime)
{
	setVecMean(meanGain1); setVecMean(meanGain2);
	printf("Gain terms of policy grad means: [%f] [%f]\n",
	meanGain1[0], meanGain2[0]);
	Learner::processStats(_stats, avgTime);
}

