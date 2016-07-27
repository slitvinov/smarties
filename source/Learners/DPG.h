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

using namespace std;

class DPG : public Learner
{
    const int nA, nS;
    Network* net_policy;
    Optimizer* opt_policy;
    void Train_BPTT(const int seq, const int first=0, const int thrID=0) override;
    void Train(const int seq, const int samp, const int first=0, const int thrID=0) override;
    
    
    void updateTargetNetwork() override;
    void stackAndUpdateNNWeights(const int nAddedGradients) override;
    void updateNNWeights(const int nAddedGradients) override;
    void allocateNNactivations(const int buffer) override;
    
public:
	DPG(Environment* env, Settings & settings);
    void select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r) override;
};

