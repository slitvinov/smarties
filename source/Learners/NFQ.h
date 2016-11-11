/*
 *  NFQ.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Learner.h"

using namespace std;

class NFQ : public Learner
{   
    Real Train_BPTT(const int seq, const int thrID=0) override;
    Real Train(const int seq, const int samp, const int thrID=0) override;
    
public:
	NFQ(Environment* env, Settings & settings);
    void select(const int agentId, State& s, Action& a, State& sOld, Action& aOld, const int info, Real r) override;
};

