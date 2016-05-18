/*
 *  TwoFish_Environment.h
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include <vector>
#include "../Util/util.h"
#include "ExternalEnvironment.h"
#include "CellList.h"

//class ExternalAgent;
//#include "../Agents/ExternalAgent.h"

class TwoFishEnvironment: public ExternalEnvironment
{
    bool sight, l_line;
    int study;
    double goalDY,gamma;
public:
    TwoFishEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, const int senses, Settings & settings);
    void setDims() override;
    bool pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward) override;
};


