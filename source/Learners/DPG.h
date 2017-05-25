/*
 *  DPG.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner.h"

class DPG : public Learner
{
    const int nA, nS;
    Network* net_policy;
    Optimizer* opt_policy;
    void Train_BPTT(const int seq, const int thrID=0) const override;
    void Train(const int seq, const int samp, const int thrID=0) const override;

    void updateTargetNetwork() override;
    void stackAndUpdateNNWeights(const int nAddedGradients) override;
    void updateNNWeights(const int nAddedGradients) override;

public:
	DPG(MPI_Comm comm, Environment*const env, Settings & settings);
    void select(const int agentId, State& s, Action& a, State& sOld,
                Action& aOld, const int info, Real r) override;
};
