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
#include "../Math/Lognormal_policy.h"

class GAE : public Learner_utils
{
	const Uint nA;
	const Real lambda = 0.99;
	std::vector<std::mt19937>& generators;
#ifdef INTEGRATEANDFIRESHARED
	vector<Uint> net_outputs = {1, nA, 1};
#else
	vector<Uint> net_outputs = {1, nA, nA};
#endif
	vector<Uint> net_indices = {0,  1, 1+nA};

	inline Lognormal_policy prepare_policy(const vector<Real>& out) const
	{
		return Lognormal_policy(net_indices[2], net_indices[3], nA, out);
	}

	void Train_BPTT(const Uint seq, const Uint thrID) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

	inline vector<Real> compute(const Uint seq, const Uint samp, Real& A_GAE, Real& Vnext, Real& V_MC, const vector<Real>& out, const Uint thrID) const
	{
		const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
		const Tuple * const t_ = data->Set[seq]->tuples[samp+1]; //contains r, sNew
		const Real V_curr = out[net_indices[0]];
		const Lognormal_policy pol = prepare_policy(out);
		//if terminal state was reached then this is r_end, and Vnext==A_GAE==V_MC=0
		A_GAE = t_->r +gamma*Vnext -V_curr +gamma*lambda*A_GAE;
		V_MC  = t_->r +gamma*V_MC;
		Vnext = V_curr; //update for previous state which will be processed next

		const Real Verr = V_MC - V_curr;
		const vector<Real> pgrad = pol.policy_grad(_t->a, A_GAE);

		vector<Real> gradient(nOutputs,0);
		gradient[net_indices[0]] = Verr;
		pol.finalize_grad(pgrad, gradient);

		//bookkeeping:
		dumpStats(Vstats[thrID], V_curr, Verr);
		data->Set[seq]->tuples[samp]->SquaredError = Verr*Verr;
		return gradient;
	}

public:
	GAE(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, const Agent& agent) override;

	void buildNetwork(Network*& _net , Optimizer*& _opt, const vector<Uint> nouts, Settings& settings,
	vector<Real> weightInitFactors = vector<Real>(),
	const vector<Uint> addedInputs = vector<Uint>()) override;

	static Uint getnOutputs(const Uint NA)
	{
#ifdef INTEGRATEANDFIRESHARED
		return 2+NA;
#else
		return 1+NA+NA;
#endif
	}
};
