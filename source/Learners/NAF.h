/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner.h"

class NAF : public Learner
{
	const Uint nA, nL;

	void Train_BPTT(const Uint seq, const Uint thrID=0) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID=0) const override;
	void dumpNetworkInfo(const int agentId);
	vector<Real> computeQandGrad(vector<Real>& grad, const vector<Real>& act,
			const vector<Real>& out, const Real Vnext) const;

public:
	NAF(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, State& s,Action& a, State& sOld,
			Action& aOld, const int info, Real r) override;
};
