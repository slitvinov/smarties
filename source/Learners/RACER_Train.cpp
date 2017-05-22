/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

void RACER::Train(const int seq, const int samp, const int thrID) const
{
		//this should go to gamma rather quick:
		const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
		const Real rGamma = annealedGamma();

		assert(net->allocatedFrozenWeights && bTrain);
		const int ndata = data->Set[seq]->tuples.size();
		assert(samp<ndata-1);
		const bool bEnd = data->Set[seq]->ended;
		const int npredicts = bEnd ? ndata-1 : ndata;
		const int nhats = npredicts - samp;
		vector<vector<Real>> out_cur(1, vector<Real>(nOutputs,0));
		vector<vector<Real>> out_hat(nhats, vector<Real>(nOutputs,0));
		vector<Activation*> series_cur = net->allocateUnrolledActivations(1);
		vector<Activation*> series_hat = net->allocateUnrolledActivations(nhats);

		for (int k=0; k<nhats; k++) {
			const Tuple * const _t = data->Set[seq]->tuples[k+samp]; //this tuple contains s, a, mu
			const vector<Real> inp = data->standardize(_t->s);
			//const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
			if(!k)
				net->seqPredict_inputs(scaledSold, series_cur[k]);
			net->seqPredict_inputs(scaledSold, series_hat[k]);
		}
		net->seqPredict_execute(series_cur, series_cur);
		net->seqPredict_execute(series_hat, series_hat, net->tgt_weights, net->tgt_biases);
		for (int k=0; k<nhats; k++) {
			if(!k)
				net->seqPredict_output(out_cur[k], series_cur[k]);
			net->seqPredict_output(out_hat[k], series_hat[k]);
		}

		Real Q_RET = 0, Q_OPC = 0;
		//if partial sequence then compute value of last state (=! R_end)
		if(not bEnd) {
			assert(data->Set[seq]->tuples[ndata-1]->mu.size() == 2*nA);
			Q_RET = Q_OPC = out_hat[nhats-1][0]; //V(s_T) computed with tgt weights
		} else assert(data->Set[seq]->tuples[ndata-1]->mu.size() == 0);

		for (int k=ndata-samp-2; k>0; k--) //just propagate Q_RET / Q_OPC to k=0
		{
			const Tuple * const _t = data->Set[seq]->tuples[k+samp]; //this tuple contains sOld, a
			const Tuple * const t_ = data->Set[seq]->tuples[k+1+samp]; //this contains r, sNew
			Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
			Q_OPC = t_->r + rGamma*Q_OPC;
			//get everybody camera ready:
			//mean of the stochastic policy:
			const vector<Real> polHat = extractPolicy(out_hat[k]);
			//pos def matrix for quadratic Q:
			const vector<Real> P_Hat = preparePmatrix(out_hat[k]);

			#ifndef ACER_RELAX
				//location of max of quadratic Q
				const vector<Real> mu_Hat = extractQmean(out_hat[k]);
			#else
				const vector<Real> mu_Hat = polHat;
			#endif

			#ifndef ACER_SAFE
				//pass through softplus to make it pos def:
				const vector<Real> preHat = extractPrecision(out_hat[k]);
				const vector<Real> varHat = extractVariance(out_hat[k]);
			#else
				const vector<Real> preHat = vector<Real>(nA, precision);
				const vector<Real> varHat = vector<Real>(nA, variance);
			#endif

			//off policy stored action and on-policy sample:
			const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
			const Real actProbOnTarget = evaluateLogProbability(act, polHat, preHat);
			const Real actProbBehavior = evaluateLogBehavioralPolicy(act, _t->mu);
			const Real rho_hat = safeExp(actProbOnTarget-actProbBehavior);
			const Real c_hat = std::min((Real)1.,std::pow(rho_hat,1./nA));

			//compute Q using tgt net for pi and C, for consistency of derivatives
			//Q(s,a)                     v a	v policy    v quadratic Q parameters
			const Real A_hat = computeAdvantage(act, polHat, varHat, P_Hat, mu_Hat);
			//prepare rolled Q with off policy corrections for next step:
			Q_RET = c_hat*1.*(Q_RET -A_hat -out_hat[k][0]) +out_hat[k][0];
			//TODO: now Q_OPC ios actually Q_RET, which is better?
			//Q_OPC = c_cur*1.*(Q_OPC -A_hat -out_hat[k][0]) +Vs;
			Q_OPC = 0.5*(Q_OPC -A_hat -out_hat[k][0]) +out_hat[k][0];
		}

		{
			const Real k = 0; ///just to make it easier to check with BPTT
			const Tuple * const _t = data->Set[seq]->tuples[samp]; //this tuple contains sOld, a
			const Tuple * const t_ = data->Set[seq]->tuples[samp+1]; //this contains r, sNew
			Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
			Q_OPC = t_->r + rGamma*Q_OPC;
			//get everybody camera ready:
			//mean of the stochastic policy:
			const vector<Real> polCur = extractPolicy(out_cur[k]);
			const vector<Real> polHat = extractPolicy(out_hat[k]);
			//pos def matrix for quadratic Q:
			const vector<Real> P_Cur = preparePmatrix(out_cur[k]);
			const vector<Real> P_Hat = preparePmatrix(out_hat[k]);

			#ifndef ACER_RELAX
				//location of max of quadratic Q
				const vector<Real> mu_Cur = extractQmean(out_cur[k]);
				const vector<Real> mu_Hat = extractQmean(out_hat[k]);
			#else
				const vector<Real> mu_Cur = polCur;
				const vector<Real> mu_Hat = polHat;
			#endif

			#ifndef ACER_SAFE
				//pass through softplus to make it pos def:
				const vector<Real> preCur = extractPrecision(out_cur[k]);
				const vector<Real> preHat = extractPrecision(out_hat[k]);
				const vector<Real> varCur = extractVariance(out_cur[k]);
				const vector<Real> varHat = extractVariance(out_hat[k]);
			#else
				const vector<Real> preCur = vector<Real>(nA, precision);
				const vector<Real> preHat = vector<Real>(nA, precision);
				const vector<Real> varCur = vector<Real>(nA, variance);
				const vector<Real> varHat = vector<Real>(nA, variance);
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

			//compute quantities needed for trunc import sampl with bias correction
			const Real importance = std::min(rho_cur, truncation);
			const Real correction = std::max(0., 1.-truncation/rho_pol);
			const Real A_OPC = Q_OPC - out_hat[k][0];

			#ifdef ACER_VARIATE
				const Real cov_A_A = out_cur[k][nOutputs-1];
				//const vector<Real> smp = samplePolicy(polCur, varCur, thrID);
				const Real varCritic = advantageVariance(polCur, varCur, P_Hat, mu_Hat);
				//const Real A_tgt = computeAdvantage(smp, polCur, varCur, P_Hat, mu_Hat);
				const Real A_cov = computeAdvantage(act, polCur, varCur, P_Hat, mu_Hat);
				const Real err_Cov = A_OPC*A_cov - cov_A_A;
				//const Real cotrolVar = A_tgt;
				const Real cotrolVar = A_cov;
				//const Real cotrolVar = nA+diagTerm(varCur,polCur,mu_Hat)
				//												 -diagTerm(varCur,   pol,mu_Hat);

				const Real eta = anneal*std::min(std::max(-.5, cov_A_A/varCritic), 0.5);
				//const Real eta = 0;
			#else
				#ifdef ACER_PENALIZER
					const Real varCritic = advantageVariance(polCur, varCur, P_Hat, mu_Hat);
					const Real A_cov = computeAdvantage(act, polCur, varCur, P_Hat, mu_Hat);
					static const Real L = 0.5, eps = 2.2e-16;
					//static const Real L = 2.2e-16, eps = 2.2e-16;
					const Real threshold = A_cov * A_cov / (varCritic+eps);
					const Real smoothing = threshold>L ? L/(threshold+eps) : 2-threshold/L;
					const Real eta = anneal * smoothing * A_cov * A_OPC / (varCritic+eps);
					//eta = eta > 1 ? 1 : (eta < -1 ? -1 : eta);
					const Real cotrolVar = A_cov, err_Cov = 0;
				#else
					const Real eta = 0, cotrolVar = 0, err_Cov = 0;
				#endif
			#endif

			const Real gain1 = A_OPC * importance - eta * rho_cur * cotrolVar;
			//const Real gain1 = A_OPC * importance - eta * cotrolVar;
			const Real gain2 = A_pol * correction;

			//derivative wrt to statistics
			const vector<Real> gradAcer_1 = policyGradient(polCur, preCur, act, gain1);
			const vector<Real> gradAcer_2 = policyGradient(polCur, preCur, pol, gain2);

			#if defined(ACER_VARIATE) || defined(ACER_PENALIZER)
			const vector<Real> gradC = controlGradient(polCur, varCur, P_Hat, mu_Hat, eta);
			const vector<Real> policy_grad = sum3Grads(gradAcer_1, gradAcer_2, gradC);
			#else
			const vector<Real> policy_grad = sum2Grads(gradAcer_1, gradAcer_2);
			#endif

			//trust region updating
			const vector<Real> gradDivKL = gradDKL(polCur, polHat, preCur, preHat);
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

			const vector<Real> critic_grad =
			criticGradient(P_Cur, polHat, varHat, out_cur[k], mu_Cur, act, Qer);
			const vector<Real> grad =
			finalizeGradient(Ver, critic_grad, policy_grad, out_cur[k], err_Cov);
			//write gradient onto output layer
			net->setOutputDeltas(grad, series_cur[k]);
	      //printf("Applying gradient %s\n",printVec(grad).c_str());
	      //fflush(0);
	      //
			//bookkeeping:
	      //#ifndef NDEBUG
	      		vector<Real> _dump = grad; _dump.push_back(gain1); _dump.push_back(eta);
			statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], _dump);
	      //#endif
			vector<Real> fake{A_cur, 100};
			dumpStats(Vstats[thrID], A_cur+out_cur[k][0], Qer, fake);
			if(thrID == 1) net->updateRunning(series_cur[k]);
			data->Set[seq]->tuples[samp]->SquaredError = Qer*Qer;
			//data->Set[seq]->tuples[k]->SquaredError = std::pow(A_OPC*rho_cur,2);
		}

		if (thrID==0) net->backProp(series_cur, net->grad);
		else net->backProp(series_cur, net->Vgrad[thrID]);
		net->deallocateUnrolledActivations(&series_cur);
		net->deallocateUnrolledActivations(&series_hat);
}
