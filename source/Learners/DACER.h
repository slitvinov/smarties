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
	const Real truncation, delta;
	std::vector<std::mt19937>& generators;
	const vector<Uint> net_outputs = {1, nA, nA};
	const vector<Uint> net_indices = {0,  1, 1+nA};

	void Train_BPTT(const Uint seq, const Uint thrID=0) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID=0) const override;

	inline Discrete_policy prepare_policy(const vector<Real>& out) const
	{
		return Discrete_policy(net_indices[1], net_indices[2], nA, out);
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
