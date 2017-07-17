/*
 *  NFQ.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner_utils.h"

using namespace std;

class NFQ : public Learner_utils
{
	void Train_BPTT(const Uint seq, const Uint thrID) const override;
	void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

public:
	NFQ(MPI_Comm comm, Environment*const env, Settings & settings);
	void select(const int agentId, const Agent& agent) const override;
};
