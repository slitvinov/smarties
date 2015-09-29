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

double ExternalAgent::getReward()
{
    return r;
}

double ExternalAgent::altReward()
{
    return _r;
}

void ExternalAgent::getState(State& _s)
{
    _s = *s;
}

void ExternalAgent::act(Action& _a)
{
    *a = _a;
}
