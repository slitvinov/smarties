/*
 *  TwoFish_Environment.h
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include "../Util/util.h"
#include "Environment.h"

class TwoActFishEnvironment: public Environment
{
protected:
    const bool sight, rcast, lline, press;
    const int study;
    const Real goalDY;
public:
    TwoActFishEnvironment(const int nAgents, const string execpath,
                          const int _rank, Settings & settings);
    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a,
                    const State& t_sN, Real& reward, const int info) override;


    void setAction(const int & iAgent) override;
    int getState(int & iAgent) override;
};
