/*
 *  Environment.h
 *  rl
 *
 *  Created by Guido Novati on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Environment.h"

class CMAEnvironment : public Environment
{
public:
	CMAEnvironment(const int nAgents, const string execpath,
								 const int _rank, Settings & settings);
    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a,
										const State& t_sN, Real& reward, const int info) override;
};
