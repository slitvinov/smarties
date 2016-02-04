/*
 *  CartAgent.cpp
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "HardCartAgent.h"


void HardCartAgent::setEnvironment(Environment* env)
{
    this->env = static_cast<HardCartEnvironment*> (env);
    environment = env;
}

Real HardCartAgent::getReward()
{
    return r;
}

void HardCartAgent::getState(State& _s)
{
    _s = *s;
}

void HardCartAgent::act(Action& _a)
{
    *a = _a;
}
