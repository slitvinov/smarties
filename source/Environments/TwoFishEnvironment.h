/*
 *  TwoFish_Environment.h
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include "Environment.h"

class TwoFishEnvironment: public Environment
{
protected:
    bool sight, l_line;
    int study;
    Real goalDY, gamma;
public:
    TwoFishEnvironment(const int nAgents, const string execpath, const int _rank, Settings & settings);
    void setDims() override;
    bool pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward) override;
};


