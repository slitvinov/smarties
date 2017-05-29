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
DiscreteAlgorithm(comm,_env,settings, 1), truncation(100), cntGrad(nThreads+1,0),
stdGrad(nThreads+1,vector<Real>(nOutputs+2,0)),
avgGrad(nThreads+1,vector<Real>(nOutputs+2,0))
{
	const vector<Uint> noutputs = {1,nA,nA};
	assert(nOutputs == 1+nA+nA);
	buildNetwork(net, opt, noutputs, settings);
	assert(nOutputs == net->getnOutputs());
	assert(nInputs == net->getnInputs());
	data->bRecurrent = bRecurrent = true;
	checkGradient();
}

void DACER::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	vector<Real> output = basicNetOut(agentId, s, a, sOld, aOld, info, r);
	if (output.size() == 0) return;
	assert(output.size() == nOutputs);
	//variance is pos def: transform linear output layer with softplus
	const vector<Real> pol = extractPolicy(output);
	const vector<Real> beta = basicNetOut(a, pol);
	assert(beta.size() == nA);
	data->passData(agentId, info, sOld, a, beta, s, r);
}

void DACER::Train(const Uint seq, const Uint samp, const Uint thrID) const { }

void DACER::Train_BPTT(const Uint seq, const Uint thrID) const
{
	//this should go to gamma rather quick:
	const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
	const Real rGamma = annealedGamma();

	assert(net->allocatedFrozenWeights && bTrain);
	const Uint ndata = data->Set[seq]->tuples.size();
	vector<vector<Real>> out_cur(ndata-1, vector<Real>(nOutputs,0));
	vector<vector<Real>> out_hat(ndata-1, vector<Real>(nOutputs,0));
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
	for (Uint k=0; k<ndata-1; k++) {
		net->seqPredict_output(out_cur[k], series_cur[k]);
		net->seqPredict_output(out_hat[k], series_hat[k]);
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
		else assert(data->Set[seq]->tuples[ndata-1]->mu.size() == 0);
	#endif

	for (int k=static_cast<int>(ndata)-2; k>=0; k--)
	{
		const Tuple * const _t = data->Set[seq]->tuples[k]; //this tuple contains sOld, a
		const Tuple * const t_ = data->Set[seq]->tuples[k+1]; //this contains r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		//get everybody camera ready:
		const vector<Real> valCur = extractValues(out_cur[k]);
		const vector<Real> valHat = extractValues(out_hat[k]);
		const vector<Real> polCur = extractPolicy(out_cur[k]);
		const vector<Real> polHat = extractPolicy(out_hat[k]);
		//off policy stored action and on-policy sample:
		const Uint act = aInfo.actionToLabel(_t->a); //unbounded action space
		const Uint pol = samplePolicy(polCur, thrID);

		const Real rho_cur = polCur[act]/_t->mu[act];
		const Real rho_pol = polCur[pol]/_t->mu[pol];
		const Real rho_hat = polHat[act]/_t->mu[act];
		const Real c_cur = std::min((Real)1.,rho_cur);
		const Real c_hat = std::min((Real)1.,rho_hat);

		//compute Q using tgt net for pi and C, for consistency of derivatives
		const Real A_cur = computeAdvantage(act, polHat, valHat);
		const Real A_hat = computeAdvantage(act, polHat, valHat);
		const Real A_pol = computeAdvantage(pol, polCur, valCur);
		//compute quantities needed for trunc import sampl with bias correction
		const Real importance = std::min(rho_cur, truncation);
		const Real correction = std::max(0., 1.-truncation/rho_pol);
		const Real A_OPC = Q_OPC - out_hat[k][0];

		#ifdef ACER_PENALIZER
			#warning "Wrong analytic derivatives for ACER_PENALIZER with DACER"
			const Real varCritic = advantageVariance(polCur, valHat);
			const Real A_cov = computeAdvantage(act, polCur, valHat);
			static const Real L = 0.25, eps = 2.2e-16;
			//static const Real L = 2.2e-16, eps = 2.2e-16;
			const Real threshold = A_cov * A_cov / (varCritic+eps);
			const Real smoothing = threshold>L ? L/(threshold+eps) : 2-threshold/L;
			const Real eta = anneal * smoothing * A_cov * A_OPC / (varCritic+eps);
			//eta = eta > 1 ? 1 : (eta < -1 ? -1 : eta);
			const Real cotrolVar = A_cov, err_Cov = 0;
		#else
			const Real eta = 0, cotrolVar = 0, err_Cov = 0;
		#endif

		const Real gain1 = A_OPC * importance - eta * rho_cur * cotrolVar;
		//const Real gain1 = A_OPC * importance - eta * cotrolVar;
		const Real gain2 = A_pol * correction;

		//derivative wrt to statistics
		const vector<Real> gradAcer_1 = policyGradient(out_cur[k], polCur, act, gain1);
		const vector<Real> gradAcer_2 = policyGradient(out_cur[k], polCur, pol, gain2);

		#ifdef ACER_PENALIZER
		const vector<Real> gradC = controlGradient(act, polCur, valHat, eta);
		const vector<Real> policy_grad = sum3Grads(gradAcer_1, gradAcer_2, gradC);
		#else
		const vector<Real> policy_grad = sum2Grads(gradAcer_1, gradAcer_2);
		#endif

		//trust region updating
		const vector<Real> gradDivKL = gradDKL(out_cur[k],out_hat[k], polCur, polHat);
		const vector<Real> gradAcer = gradAcerTrpo(policy_grad, gradDivKL);
		const Real Vs  = stateValue(out_cur[k][0],out_hat[k][0]);
		const Real Qer = (Q_RET -A_cur -out_cur[k][0]);
		//unclear usefulness:
		//const Real Ver = (Q_RET -A_cur -out_cur[k][0])*std::min(1.,rho_hat);
		const Real Ver = 0;
		//prepare rolled Q with off policy corrections for next step:
		Q_RET = c_hat*1.*(Q_RET -A_hat -out_hat[k][0]) +Vs;
		//TODO: now Q_OPC ios actually Q_RET, which is better?
		//Q_OPC = c_cur*1.*(Q_OPC -A_hat -out_hat[k][0]) +Vs;
		Q_OPC = 0.5*(Q_OPC -A_hat -out_hat[k][0]) +Vs;

		const vector<Real> critic_grad = criticGradient(act, polHat, valCur, Qer);
		const vector<Real> grad = finalizeGradient(Ver, critic_grad, policy_grad);
		net->setOutputDeltas(grad, series_cur[k]);

		vector<Real> _dump = grad; _dump.push_back(gain1); _dump.push_back(eta);
		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], _dump);
      //#endif
		vector<Real> fake{A_cur, 100};
		dumpStats(Vstats[thrID], A_cur+out_cur[k][0], Qer, fake);
		data->Set[seq]->tuples[k]->SquaredError = Qer*Qer;
		//data->Set[seq]->tuples[k]->SquaredError = std::pow(A_OPC*rho_cur,2);
	}

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	net->deallocateUnrolledActivations(&series_hat);
}

void DACER::processStats(vector<trainData*> _stats, const Real avgTime)
{
	#ifdef ACER_SAFE
		const Real stdev = 0.1 + annealingFactor();
		variance = stdev*stdev;
		precision = 1./variance;
	#endif
   //#ifndef NDEBUG
	statsVector(avgGrad, stdGrad, cntGrad);
	//setVecMean(meanGain1); setVecMean(meanGain2);
	printf("Avg grad [%s] - std [%s]\n",
	printVec(avgGrad[0]).c_str(), printVec(stdGrad[0]).c_str());
	fflush(0);
   //#endif
	Learner::processStats(_stats, avgTime);
}

void DACER::dumpPolicy(const vector<Real> lower, const vector<Real>& upper,
		const vector<Uint>& nbins)
 {
	 /*
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

		#ifndef ACER_SAFE
			vector<Real> var =  extractVariance(output);
			dump[cnt++] = std::sqrt(var[0]);
		#else
			dump[cnt++] = std::sqrt(variance);
		#endif

		#ifndef ACER_RELAX
			vector<Real> mean = extractQmean(output);
			dump[cnt++] = aInfo.getScaled(mean[0], 0);
		#else
			dump[cnt++] = aInfo.getScaled(mu[0], 0);
		#endif

		for (int j=0; j<state.size(); j++) dump[cnt++] = state[j];
		assert(cnt == dump.size());
		fwrite(dump.data(),sizeof(Real),state.size()+4,pFile);
	}
	fclose (pFile);
	*/
 }
