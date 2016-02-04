/*
 *  ExternalAgent.cpp
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "ExternalAgent.h"


void ExternalAgent::setEnvironment(Environment* env)
{
    this->env = static_cast<ExternalEnvironment*> (env);
    environment = env;
}

Real ExternalAgent::getReward()
{
    return r;
}

Real ExternalAgent::getInfo(int n)
{
    if (n<nInfo)
    return Info[n];
    else
    return 0;
}

void ExternalAgent::getState(State& _s)
{
    _s = *s;
}

void ExternalAgent::act(Action& _a)
{
    *a = _a;
}
