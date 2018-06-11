//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the “CC BY-SA 4.0” license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

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
};
