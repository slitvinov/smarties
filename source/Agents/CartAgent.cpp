/*
 *  CartAgent.cpp
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "CartAgent.h"


void CartAgent::setEnvironment(Environment* env)
{
    this->env = static_cast<CartEnvironment*> (env);
    environment = env;
}

double CartAgent::getReward()
{
    return r;
}

void CartAgent::getState(State& _s)
{
    _s = *s;
}

void CartAgent::act(Action& _a)
{
    *a = _a;
}
