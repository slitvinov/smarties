/*
 *  DeadFish_Environment.h
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include "Environment.h"

class DeadFishEnvironment: public Environment
{
protected:
    const bool sight, rcast, lline, press;
    const int study;
public:
    DeadFishEnvironment(const Uint nAgents, const string execpath, Settings & settings);
    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a,
                    const State& t_sN, Real& reward, const int info) override;
};
