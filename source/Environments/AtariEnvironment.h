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

class AtariEnvironment: public Environment
{
public:
    AtariEnvironment(Settings & _settings);
    bool predefinedNetwork(Builder & input_net) const override;
};
