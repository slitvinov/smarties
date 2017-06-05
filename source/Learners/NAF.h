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
#include "../Math/Quadratic_advantage.h"
#include "../Math/FeatureControlTasks.h"

class NAF : public Learner_utils
{
	const Uint nA, nL;
	//Network produces a vector. The two following vectors specify:
	// - the sizes of the elements that compose the vector
	// - the starting indices along the output vector of each
	vector<Uint> net_outputs = {1, nL, nA};
	vector<Uint> net_indices = {0, 1, 1+nL};
	#ifdef FEAT_CONTROL
	const ContinuousFeatureControl* task;
	#endif

	inline Quadratic_advantage prepare_advantage(const vector<Real>& out) const
	{
		return Quadratic_advantage(net_indices[1], net_indices[2], nA, nL, out);
	}

	void Train_BPTT(const Uint seq, const Uint thrID=0) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID=0) const override;

public:
	NAF(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, State& s,Action& a, State& sOld,
			Action& aOld, const int info, Real r) override;
	void test();
	static inline Uint compute_nL(const Uint NA)
	{
		return (NA*NA + NA)/2;
	}
};
