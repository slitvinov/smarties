/*
 *  DPG.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner_utils.h"

class DPG : public Learner_utils
{
	const Uint nA, nS;
	mutable vector<long double> cntValGrad;
	mutable vector<vector<long double>> avgValGrad, stdValGrad;
	Network* net_value;
	Optimizer* opt_value;

	void Train_BPTT(const Uint seq, const Uint thrID=0) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID=0) const override;

	void updateTargetNetwork() override;
	void stackAndUpdateNNWeights(const Uint nAddedGradients) override;
	void processGrads() override;

public:
	DPG(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, State& s, Action& a, State& sOld,
			Action& aOld, const int info, Real r) override;
};
