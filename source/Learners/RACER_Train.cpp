/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

void RACER::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
	//this should go to gamma rather quick:
	const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
	const Real rGamma = annealedGamma();

	const Uint ndata = data->Set[seq]->tuples.size();
	assert(samp<ndata-1);
	const bool bEnd = data->Set[seq]->ended;
	const Uint nMaxTargets = MAX_UNROLL_AFTER+1;
	//for off policy correction we need reward and action, therefore not last one:
	const Uint nSUnroll = min(ndata-1-samp, nMaxTargets);
	//if we do not have a terminal reward, then we compute value of last state:
	const Uint nSValues = min(bEnd? ndata-1-samp :ndata-samp, nMaxTargets);
	//to prevent silly overflow on aux tasks:
	const Uint nSalloc = max(nSValues, static_cast<Uint>(2));
	vector<vector<Real>> out_cur(1, vector<Real>(nOutputs,0));
	vector<vector<Real>> out_hat(nSValues, vector<Real>(nOutputs,0));
	vector<Activation*> series_cur = net->allocateUnrolledActivations(1);
	vector<Activation*> series_hat = net->allocateUnrolledActivations(nSalloc);
	//printf("%d %u %u %u %u %u \n", bEnd, samp, ndata, nSUnroll, nSValues, nSalloc); fflush(0);
	for (Uint k=0; k<nSValues; k++) {
		const Tuple * const _t = data->Set[seq]->tuples[k+samp]; //this tuple contains s, a, mu
		const vector<Real> inp = data->standardize(_t->s);
		//const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
		if(!k) net->seqPredict_inputs(inp, series_cur[k]);
		net->seqPredict_inputs(inp, series_hat[k]);
	}
	net->seqPredict_execute(series_cur, series_cur);
	net->seqPredict_execute(series_hat, series_hat, net->tgt_weights, net->tgt_biases);
	for (Uint k=0; k<nSValues; k++) {
		if(!k) net->seqPredict_output(out_cur[k], series_cur[k]);
		net->seqPredict_output(out_hat[k], series_hat[k]);
	}

	Real Q_RET = 0, Q_OPC = 0;
	//if partial sequence then compute value of last state (=! R_end)
	if(nSValues != nSUnroll) {
		assert(nSValues>nSUnroll && !bEnd);
		Q_RET=Q_OPC= out_hat[nSValues-1][net_indices[0]]; //V(s_T) with tgt weights
	} else assert(data->Set[seq]->tuples[ndata-1]->mu.size() == 0);

	for (int k=static_cast<int>(nSUnroll)-1; k>0; k--) //propagate Q to k=0
	{
		const Tuple*const _t=data->Set[seq]->tuples[k+samp]; //contains sOld, a
		const Tuple*const t_=data->Set[seq]->tuples[k+1+samp]; //contains r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		const Real V_hat = out_hat[k][net_indices[0]];
		const Gaussian_policy pol_hat = prepare_policy(out_hat[k]);
		//Used as target: target policy, target value
		const Quadratic_advantage adv_hat = prepare_advantage(out_hat[k], &pol_hat);
		//off policy stored action:
		const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
		const Real actProbOnTarget = pol_hat.evalLogProbability(act);
		const Real actProbBehavior = Gaussian_policy::evalBehavior(act,_t->mu);
		const Real rho_hat = safeExp(actProbOnTarget-actProbBehavior);
		const Real c_hat = std::min((Real)1.,std::pow(rho_hat,1./nA));
		const Real A_hat = adv_hat.computeAdvantage(act);
		const Real lambda = 0.5;
		//const Real lambda = 1.0;
		//prepare rolled Q with off policy corrections for next step:
		Q_RET = c_hat*lambda*(Q_RET -A_hat -V_hat) +V_hat;
		Q_OPC =       lambda*(Q_OPC -A_hat -V_hat) +V_hat;
		//Q_OPC = Q_RET;
	}

	{
		const Uint k = 0; ///just to make it easier to check with BPTT
		const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
		const Tuple * const t_ = data->Set[seq]->tuples[samp+1]; //contains r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		//get everybody camera ready:
		const Real V_cur = out_cur[k][net_indices[0]];
		const Real V_hat = out_hat[k][net_indices[0]];
		const Gaussian_policy pol_cur = prepare_policy(out_cur[k]);
		const Gaussian_policy pol_hat = prepare_policy(out_hat[k]);
		//Used for update of value: target policy, current value
		const Quadratic_advantage adv_cur = prepare_advantage(out_cur[k], &pol_hat);
		//Used as target: target policy, target value
		const Quadratic_advantage adv_hat = prepare_advantage(out_hat[k], &pol_hat);
		//Used for update of policy: current policy, target value
		const Quadratic_advantage adv_pol = prepare_advantage(out_hat[k], &pol_cur);

		//off policy stored action and on-policy sample:
		const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
		const vector<Real> pol = pol_cur.sample(&generators[thrID]);

		const Real actProbOnPolicy = pol_cur.evalLogProbability(act);
		const Real polProbOnPolicy = pol_cur.evalLogProbability(pol);
		const Real actProbOnTarget = pol_hat.evalLogProbability(act);
		const Real actProbBehavior = Gaussian_policy::evalBehavior(act,_t->mu);
		const Real polProbBehavior = Gaussian_policy::evalBehavior(pol,_t->mu);
		const Real rho_cur = safeExp(actProbOnPolicy-actProbBehavior);
		const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
		const Real rho_hat = safeExp(actProbOnTarget-actProbBehavior);
		const Real c_cur = std::min((Real)1.,std::pow(rho_cur,1./nA));
		const Real c_hat = std::min((Real)1.,std::pow(rho_hat,1./nA));
		const Real varCritic = adv_pol.advantageVariance();
		const Real A_cur = adv_cur.computeAdvantage(act);
		const Real A_hat = adv_hat.computeAdvantage(act);
		const Real A_pol = adv_pol.computeAdvantage(pol);
		const Real A_cov = adv_pol.computeAdvantage(act);
		//compute quantities needed for trunc import sampl with bias correction
		const Real importance = std::min(rho_cur, truncation);
		const Real correction = std::max(0., 1.-truncation/rho_pol);
		const Real A_OPC = Q_OPC - V_hat;
		static const Real L = 0.25, eps = 2.2e-16;
		const Real threshold = A_cov * A_cov / (varCritic+eps);
		const Real smoothing = threshold>L ? L/(threshold+eps) : 2-threshold/L;
		const Real eta = anneal * smoothing * A_cov * A_OPC / (varCritic+eps);

		#ifdef ACER_PENALIZER
		const Real cotrolVar = A_cov;
		#else
		const Real cotrolVar = 0;
		#endif
		const Real gain1 = A_OPC * importance - eta * rho_cur * cotrolVar;
		const Real gain2 = A_pol * correction;
		const vector<Real> gradAcer_1 = pol_cur.policy_grad(act, gain1);
		const vector<Real> gradAcer_2 = pol_cur.policy_grad(pol, gain2);
		#ifdef ACER_PENALIZER
		const vector<Real> gradC = pol_cur.control_grad(&adv_pol, eta);
		const vector<Real> policy_grad = sum3Grads(gradAcer_1, gradAcer_2, gradC);
		#else
		const vector<Real> policy_grad = sum2Grads(gradAcer_1, gradAcer_2);
		#endif
		//trust region updating
		const vector<Real> gradDivKL = pol_cur.div_kl_grad(&pol_hat);
		const vector<Real> trust_grad=
			trust_region_update(policy_grad,gradDivKL,delta);
		const Real Qer = (Q_RET -A_cur -V_cur);

		vector<Real> gradient(nOutputs,0);
		gradient[net_indices[0]]= Qer;
		adv_cur.grad(act, Qer, gradient);
		pol_cur.finalize_grad(trust_grad, gradient);

		#ifdef FEAT_CONTROL
		task->Train(series_cur[k],series_hat[k+1],act,seq,samp,rGamma,gradient);
		#endif

		//bookkeeping:
		dumpStats(Vstats[thrID], A_cur+V_cur, Qer);
		data->Set[seq]->tuples[samp]->SquaredError = Qer*Qer;
		vector<Real> _dump = gradient; _dump.push_back(gain1); _dump.push_back(eta);
		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], _dump);

		//write gradient onto output layer:
		clip_gradient(gradient, stdGrad[0]);
		net->setOutputDeltas(gradient, series_cur[k]);
	}

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	net->deallocateUnrolledActivations(&series_hat);
}
