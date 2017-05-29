/*
 *  Environment.h
 *  rl
 *
 *  Created by Guido Novati, modified by Iveta Rott on January 7th, 2017
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Environment.h"

class AcrobotEnvironment : public Environment
{
public:
    AcrobotEnvironment(const Uint nAgents, const string execpath, Settings & settings);

    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a,
                    const State& t_sN, Real& reward, const int info) override;
};
