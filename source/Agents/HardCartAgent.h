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
class HardCartAgent : public Agent
{
public:
    using Agent::Agent;

    State *s;
    Action *a;
    Real r;

    HardCartEnvironment* env;

    void   getState(State& s);
    Real getReward();
    void   act(Action& a);
    void   move(Real dt) {};

    void   setEnvironment(Environment* env);
};


