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

class CartEnvironment : public Environment
{
public:
    CartEnvironment(const int nAgents, const string execpath, const int _rank, Settings & settings);

    void setDims() override;
    bool pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward) override;
	bool predefinedNetwork(Network* const net) const override;
};
