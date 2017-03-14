/*
 *  DeadFish_Environment.h
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include "../Util/util.h"
#include "Environment.h"

class DeadFishEnvironment: public Environment
{
protected:
    const bool sight, POV, l_line, p_sensors;
    const int study;
    const Real goalDY;
public:
    DeadFishEnvironment(const int nAgents, const string execpath,
                       const int _rank, Settings & settings);
    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a,
                    const State& t_sN, Real& reward, const int info) override;
};
