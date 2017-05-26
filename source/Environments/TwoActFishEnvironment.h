/*
 *  TwoFish_Environment.h
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once
//#define __DBG_CNN
#include "Environment.h"

class TwoActFishEnvironment: public Environment
{
protected:
    const bool sight, rcast, lline, press;
    const int study;
    const Real goalDY;
public:
    TwoActFishEnvironment(const Uint nAgents, const string execpath,
                          const Uint _rank, Settings & settings);
    void setDims() override;
    bool pickReward(const State& t_sO, const Action& t_a,
                    const State& t_sN, Real& reward, const int info) override;
    #ifdef __DBG_CNN
    bool predefinedNetwork(Builder* const net) const override;
    #endif
};
