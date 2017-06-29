/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Learner_utils.h"
#include "../Math/Discrete_policy.h"

class DACER : public Learner_utils
{
	const Uint nA;
	#ifdef ACER_TABC
	const Real truncation;
	#endif
	const Real delta;
	std::vector<std::mt19937>& generators;
	const vector<Uint> net_outputs = {1, nA, nA};
	const vector<Uint> net_indices = {0,  1, 1+nA};

	void Train_BPTT(const Uint seq, const Uint thrID=0) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID=0) const override;

	inline Discrete_policy prepare_policy(const vector<Real>& out) const
	{
		return Discrete_policy(net_indices[1], net_indices[2], nA, out);
	}

	template<int bUpdateOPC>
	inline vector<Real> compute(const Uint seq, const Uint samp, Real& Q_RET,
		Real& Q_OPC, const vector<Real>& out_cur, const vector<Real>& out_hat,
		const Real rGamma, const Uint thrID) const
	{
		const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
		const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
		const Tuple * const t_ = data->Set[seq]->tuples[samp+1]; //contains r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		//get everybody camera ready:
		const Real V_cur = out_cur[net_indices[0]];
		const Real V_hat = out_hat[net_indices[0]];
		const Discrete_policy pol_cur = prepare_policy(out_cur);
		const Discrete_policy pol_hat = prepare_policy(out_hat);

		//off policy stored action and on-policy sample:
		const Uint act = aInfo.actionToLabel(_t->a);
		const Real rho_cur = pol_cur.probability(act)/_t->mu[act];
		const Real varCritic = pol_cur.advantageVariance();
		const Real A_cur = pol_cur.computeAdvantage(act);
		const Real Qer = Q_RET - A_cur-V_cur, A_OPC = Q_OPC - V_hat;

		static const Real L = 0.1, eps = 2.2e-16;
		const Real threshold = A_cur * A_cur / (varCritic+eps);
		const Real smoothing = threshold>L ? L/(threshold+eps) : 2-threshold/L;
		const Real eta = anneal * smoothing * A_cur * A_OPC / (varCritic+eps);

		#ifdef ACER_PENALIZER
			const Real cotrolVar = A_cur;
		#else
			const Real cotrolVar = 0;
		#endif

		//compute quantities needed for trunc import sampl with bias correction
		#ifdef ACER_TABC
			const Uint pol = pol_cur.sample(&generators[thrID]);
			const Real rho_pol = pol_cur.probability(pol)/_t->mu[pol];
			const Real A_pol = pol_cur.computeAdvantage(pol);
			const Real gain1 = A_OPC*min(rho_cur,truncation) -eta*rho_cur*cotrolVar;
			const Real gain2 = A_pol*max(0.,1.-truncation/rho_pol);

			const vector<Real> gradAcer_1 = pol_cur.policy_grad(act, gain1);
			const vector<Real> gradAcer_2 = pol_cur.policy_grad(pol, gain2);
			const vector<Real> gradAcer = sum2Grads(gradAcer_1, gradAcer_2);
		#else
			const Real gain1 = A_OPC * rho_cur - eta * rho_cur * cotrolVar;
			const vector<Real> gradAcer = pol_cur.policy_grad(act, gain1);
		#endif

		#ifdef ACER_PENALIZER
			const vector<Real> gradC = pol_cur.control_grad(eta);
			const vector<Real> policy_grad = sum2Grads(gradAcer, gradC);
		#else
			const vector<Real> policy_grad = gradAcer;
		#endif

		//trust region updating
		const vector<Real> gradDivKL = pol_cur.div_kl_grad(&pol_hat);
		const vector<Real> trust_grad=
			trust_region_update(policy_grad,gradDivKL,delta);

		#ifdef ACER_TABC
			const Real rho_hat = pol_hat.probability(act)/_t->mu[act];
			const Real Ver = Qer*std::min(1.,rho_hat); //unclear
		#else
			const Real Ver = 0;
		#endif

		vector<Real> gradient(nOutputs,0);
		gradient[net_indices[0]]= Qer + Ver;
		pol_cur.values_grad(act, Qer, gradient);
		pol_cur.finalize_grad(trust_grad, gradient);

		if(bUpdateOPC)
		{
			#ifndef ACER_TABC
				const Real rho_hat = pol_hat.probability(act)/_t->mu[act];
			#endif
			//Used as target: target policy, target value
			const Real A_hat = pol_hat.computeAdvantage(act);
			//const Real c_cur = std::min((Real)1.,std::pow(rho_cur,1./nA));
			const Real c_hat = std::min((Real)1.,std::pow(rho_hat,1./nA));
			//prepare rolled Q with off policy corrections for next step:
			//Q_RET = c_hat * minAbsValue(Q_RET-A_hat-V_hat,Q_RET-A_cur-V_cur) +
			//								minAbsValue(V_hat,V_cur);
			Q_RET = c_hat*ACER_LAMBDA*(Q_RET -A_hat -V_hat) +V_hat;
			//Q_OPC = 		ACER_LAMBDA*(Q_OPC -A_hat -V_hat) +V_hat;
			Q_OPC = Q_RET;
		}

		//bookkeeping:
		dumpStats(Vstats[thrID], A_cur+V_cur, Qer);
		data->Set[seq]->tuples[samp]->SquaredError = Qer*Qer;
		return gradient;
	}

	inline void offPolCorrUpdate(const Uint seq, const Uint samp, Real& Q_RET,
		Real& Q_OPC, const vector<Real>& output_hat, const Real rGamma) const
	{
		const Tuple*const _t=data->Set[seq]->tuples[samp]; //contains sOld, a
		const Tuple*const t_=data->Set[seq]->tuples[samp+1]; //contains r, sNew
		Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
		Q_OPC = t_->r + rGamma*Q_OPC;
		const Real V_hat = output_hat[net_indices[0]];
		const Discrete_policy pol_hat = prepare_policy(output_hat);
		//off policy stored action:
		const Uint act = aInfo.actionToLabel(_t->a);
		const Real rho_hat = pol_hat.probability(act)/_t->mu[act];
		const Real c_hat = std::min((Real)1., std::pow(rho_hat, 1./nA));
		const Real A_hat = pol_hat.computeAdvantage(act);
		//prepare rolled Q with off policy corrections for next step:
		Q_RET = c_hat*ACER_LAMBDA*(Q_RET -A_hat -V_hat) +V_hat;
		//Q_OPC =     ACER_LAMBDA*(Q_OPC -A_hat -V_hat) +V_hat;
		Q_OPC = Q_RET;
	}

public:
	DACER(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, State& s,Action& a, State& sOld,
			Action& aOld, const int info, Real r) override;

	void test();
	static Uint getnOutputs(const Uint NA)
	{
		return 1+NA+NA;
	}
};
