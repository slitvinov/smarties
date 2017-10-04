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
  const int study;
  const bool sight, rcast, lline, press;
  const Real goalDY;
public:
    TwoActFishEnvironment(Settings & _settings);
    void setDims() override;
    //bool pickReward(const Agent& agent) override;
    #ifdef __DBG_CNN
    bool predefinedNetwork(Builder* const net) const override;
    #endif
};
