/*
 *  HingedFish_Environment.h
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Modded by SV on July 13, 2017
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include "Environment.h"

class HingedFishEnvironment: public Environment
{
protected:
    bool sight, l_line;
    int study;
    Real goalDY;
public:
    HingedFishEnvironment(const Uint nAgents, const string execpath, Settings & settings);
    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a,
                    const State& t_sN, Real& reward, const int info) override;
};
