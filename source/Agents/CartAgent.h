/*
 *  CartAgent.h
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include "../StateAction.h"
#include "Agent.h"

class CartEnvironment;
class HardCartEnvironment;
#include "../Environments/Environment.h"
#include "../Environments/CartEnvironment.h"
#include "../Environments/HardCartEnvironment.h"
class CartAgent : public Agent
{
public:
    using Agent::Agent;

    State *s;
    Action *a;
    double r;

    CartEnvironment* env;

    void   getState(State& s);
    double getReward();
    void   act(Action& a);
    void   move(double dt) {};

    void   setEnvironment(Environment* env);
};


