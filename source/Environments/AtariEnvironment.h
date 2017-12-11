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
    void predefinedNetwork(Network* &net, Optimizer* &opt) const override;
};
