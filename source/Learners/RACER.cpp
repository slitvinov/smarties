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
//#include "RACER_TrainBPTT.cpp"
#include "RACER_Train.cpp"

RACER::RACER(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm,_env,settings,settings.nnOutputs+2),
truncation(10), delta(0.1), nA(_env->aI.dim), nL(compute_nL(_env->aI.dim)),
generators(settings.generators)
{
	vector<Real> out_weight_inits = {-1, -1, settings.outWeightsPrefac};
	#ifndef ACER_SAFE
	out_weight_inits.push_back(-1);
	#endif
	#ifndef ACER_RELAX
	out_weight_inits.push_back(settings.outWeightsPrefac);
	#endif
	#ifdef FEAT_CONTROL
		const Uint task_out0 = ContinuousSignControl::addRequestedLayers(nA,
			 env->sI.dimUsed, net_indices, net_outputs, out_weight_inits);
	#endif

	buildNetwork(net, opt, net_outputs, settings, out_weight_inits);
	printf("RACER: Built network with outputs: %s %s\n",
		print(net_indices).c_str(),print(net_outputs).c_str());
	assert(nOutputs == net->getnOutputs());
	assert(nInputs == net->getnInputs());
	#ifdef FEAT_CONTROL
	task = new ContinuousSignControl(task_out0, nA, env->sI.dimUsed, net, data);
	#endif
	//data->bRecurrent =bRecurrent =true;
	//data->bRecurrent =bRecurrent =false;
	test();
}

void RACER::select(const int agentId, State& s, Action& a, State& sOld,
		Action& aOld, const int info, Real r)
{
	vector<Real> output = output_stochastic_policy(agentId, s, a, sOld, aOld, info, r);
	if (output.size() == 0) return;
	assert(output.size() == nOutputs);
	//variance is pos def: transform linear output layer with softplus

	const Gaussian_policy pol = prepare_policy(output);
	const Quadratic_advantage adv = prepare_advantage(output, &pol);
	const Real anneal = annealingFactor();
	vector<Real> beta_mean=pol.getMean(), beta_std=pol.getStdev(), beta(2*nA,0);

	if(bTrain)
	for(Uint i=0; i<nA; i++) {
		beta_std[i] = max(0.2*anneal, greedyEps + beta_std[i]);
		//beta_mean[i] = (1-anneal*anneal)*beta_mean[i];
	}

	for(Uint i=0; i<nA; i++) {
		beta[i] = beta_mean[i]; //first nA contain mean
		beta[nA+i] = 1/beta_std[i]/beta_std[i]; //next nA contain precision
		std::normal_distribution<Real> dist_cur(beta_mean[i], beta_std[i]);
		a.vals[i] = positive(greedyEps+anneal) ? dist_cur(*gen) : beta_mean[i];
	}

	//scale back to action space size:
	a.set(aInfo.getScaled(a.vals));

	#if 1
	beta.insert(beta.end(), adv.matrix.begin(), adv.matrix.end());
	beta.insert(beta.end(), adv.mean.begin(),   adv.mean.end());
	#endif
	data->passData(agentId, info, sOld, a, beta, s, r);

	dumpNetworkInfo(agentId);
}

void RACER::Train_BPTT(const Uint seq, const Uint thrID) const
{
	//this should go to gamma rather quick:
	const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
	const Real rGamma = annealedGamma();

	const Uint ndata = data->Set[seq]->tuples.size();
	vector<Activation*> series_cur = net->allocateUnrolledActivations(ndata-1);
	vector<Activation*> series_hat = net->allocateUnrolledActivations(ndata);

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
		const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
		vector<Real> out_T(nOutputs, 0), S_T = data->standardize(_t->s);//last state
		net->predict(S_T,out_T,series_hat,ndata-1,net->tgt_weights,net->tgt_biases);
		Q_OPC = Q_RET = out_T[net_indices[0]]; //V(s_T) computed with tgt weights
	}
#ifndef NDEBUG
	else assert(data->Set[seq]->tuples[ndata-1]->mu.size() == 0);
#endif

	for (int k=static_cast<int>(ndata)-2; k>=0; k--)
	{
		const Tuple * const _t = data->Set[seq]->tuples[k]; //this contains sOld, a
		const Tuple * const t_ = data->Set[seq]->tuples[k+1]; //contains r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		//get everybody camera ready:
		vector<Real> out_cur = net->getOutputs(series_cur[k]);
		vector<Real> out_hat = net->getOutputs(series_hat[k]);
		const Real V_cur = out_cur[net_indices[0]];
		const Real V_hat = out_hat[net_indices[0]];
		const Gaussian_policy pol_cur = prepare_policy(out_cur);
		const Gaussian_policy pol_hat = prepare_policy(out_hat);
		//Used for update of value: target policy, current value
		const Quadratic_advantage adv_cur = prepare_advantage(out_cur, &pol_hat);
		//Used as target: target policy, target value
		const Quadratic_advantage adv_hat = prepare_advantage(out_hat, &pol_hat);
		//Used for update of policy: current policy, target value
		const Quadratic_advantage adv_pol = prepare_advantage(out_hat, &pol_cur);

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
		//const Real c_cur = std::min((Real)1.,std::pow(rho_cur,1./nA));
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
		const vector<Real> trust_grad=trust_region_update(policy_grad,gradDivKL,delta);
		const Real Qer = (Q_RET -A_cur -V_cur);
		//const Real Ver = (Q_RET -A_cur -V_cur)*std::min(1.,rho_hat); //unclear usefulness

		//prepare rolled Q with off policy corrections for next step:
		Q_RET = c_hat * minAbsValue(Q_RET-A_hat-V_hat,Q_RET-A_cur-V_cur) +
										minAbsValue(V_hat,V_cur);
		//Q_OPC = c_cur*1.*(Q_OPC -A_hat -out_hat[k][0]) +Vs;
		Q_OPC =   0.5 * minAbsValue(Q_OPC-A_hat-V_hat,Q_OPC-A_cur-V_cur) +
										minAbsValue(V_hat,V_cur);

		vector<Real> gradient(nOutputs,0);
		gradient[net_indices[0]]= Qer;
		adv_cur.grad(act, Qer, gradient);
		pol_cur.finalize_grad(trust_grad, gradient);

		#ifdef FEAT_CONTROL
		task->Train(series_cur[k],series_hat[k+1],act,seq,k,rGamma,gradient);
		#endif

		//bookkeeping:
		dumpStats(Vstats[thrID], A_cur+V_cur, Qer);
		data->Set[seq]->tuples[k]->SquaredError =Qer*Qer;
		vector<Real> _dump = gradient; _dump.push_back(gain1); _dump.push_back(eta);
		statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], _dump);

		//write gradient onto output layer:
		clip_gradient(gradient, stdGrad[0], seq, k);
		net->setOutputDeltas(gradient, series_cur[k]);
		//TODO missing  -anneal*out[j+nA*2] on means
	}

	if (thrID==0) net->backProp(series_cur, net->grad);
	else net->backProp(series_cur, net->Vgrad[thrID]);
	net->deallocateUnrolledActivations(&series_cur);
	net->deallocateUnrolledActivations(&series_hat);
}

#if 0
void RACER::offPolCorrUpdate(const Uint seq, const Uint samp, Real& Q_RET, Real& Q_OPC, const vector<Real>& output_hat, const Real rGamma) const
{
	const Tuple*const _t=data->Set[seq]->tuples[samp]; //contains sOld, a
	const Tuple*const t_=data->Set[seq]->tuples[samp+1]; //contains r, sNew
	Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
	Q_OPC = t_->r + rGamma*Q_OPC;
	const Real V_hat = output_hat[net_indices[0]];
	const Gaussian_policy pol_hat = prepare_policy(output_hat);
	//Used as target: target policy, target value
	const Quadratic_advantage adv_hat = prepare_advantage(output_hat, &pol_hat);
	//off policy stored action:
	const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
	const Real actProbOnTarget = pol_hat.evalLogProbability(act);
	const Real actProbBehavior = Gaussian_policy::evalBehavior(act,_t->mu);
	const Real rho_hat = safeExp(actProbOnTarget-actProbBehavior);
	const Real c_hat = std::min((Real)1., std::pow(rho_hat, 1./nA));
	const Real A_hat = adv_hat.computeAdvantage(act);
	const Real lambda = 0.5;
	//const Real lambda = 1.0;
	//prepare rolled Q with off policy corrections for next step:
	Q_RET = c_hat*lambda*(Q_RET -A_hat -V_hat) +V_hat;
	Q_OPC =       lambda*(Q_OPC -A_hat -V_hat) +V_hat;
	//Q_OPC = Q_RET;
}

template<int bUpdateOPC>
void RACER::compute(const Uint seq, const Uint samp, Real& Q_RET,
	Real& Q_OPC, Activation*const act_cur, const Activation*const act_hat,
	const Real rGamma, const Uint thrID) const
{
	const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
	const Tuple * const t_ = data->Set[seq]->tuples[samp+1]; //contains r, sNew
	Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
	Q_OPC = t_->r + rGamma*Q_OPC;
	//get everybody camera ready:
	const vector<Real> out_cur = net->getOutputs(act_cur);
	const vector<Real> out_hat = net->getOutputs(act_hat);
	const Real V_cur = out_cur[net_indices[0]];
	const Real V_hat = out_hat[net_indices[0]];
	const Gaussian_policy pol_cur = prepare_policy(out_cur);
	const Gaussian_policy pol_hat = prepare_policy(out_hat);
	//Used for update of value: target policy, current value
	const Quadratic_advantage adv_cur = prepare_advantage(out_cur, &pol_hat);
	//Used as target: target policy, target value
	const Quadratic_advantage adv_hat = prepare_advantage(out_hat, &pol_hat);
	//Used for update of policy: current policy, target value
	const Quadratic_advantage adv_pol = prepare_advantage(out_hat, &pol_cur);

	//off policy stored action and on-policy sample:
	const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
	const vector<Real> pol = pol_cur.sample(&generators[thrID]);

	const Real actProbOnPolicy = pol_cur.evalLogProbability(act);
	const Real polProbOnPolicy = pol_cur.evalLogProbability(pol);
	//const Real actProbOnTarget = pol_hat.evalLogProbability(act);
	const Real actProbBehavior = Gaussian_policy::evalBehavior(act,_t->mu);
	const Real polProbBehavior = Gaussian_policy::evalBehavior(pol,_t->mu);
	const Real rho_cur = safeExp(actProbOnPolicy-actProbBehavior);
	const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
	//const Real rho_hat = safeExp(actProbOnTarget-actProbBehavior);
	//const Real c_cur = std::min((Real)1.,std::pow(rho_cur,1./nA));
	//const Real c_hat = std::min((Real)1.,std::pow(rho_hat,1./nA));
	const Real varCritic = adv_pol.advantageVariance();
	const Real A_cur = adv_cur.computeAdvantage(act);
	//const Real A_hat = adv_hat.computeAdvantage(act);
	const Real A_pol = adv_pol.computeAdvantage(pol);
	const Real A_cov = adv_pol.computeAdvantage(act);
	//compute quantities needed for trunc import sampl with bias correction
	const Real importance = std::min(rho_cur, truncation);
	const Real correction = std::max(0., 1.-truncation/rho_pol);
	const Real A_OPC = Q_OPC - V_hat;
	static const Real L = 0.1, eps = 2.2e-16;
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

	//bookkeeping:
	dumpStats(Vstats[thrID], A_cur+V_cur, Qer);
	data->Set[seq]->tuples[samp]->SquaredError = Qer*Qer;
	vector<Real> _dump = gradient; _dump.push_back(gain1); _dump.push_back(eta);
	statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], _dump);
}
#endif
