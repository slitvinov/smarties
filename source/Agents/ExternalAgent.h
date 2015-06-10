/*
 *  ExternalAgent.h
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include "../StateAction.h"
#include "Agent.h"

class ExternalEnvironment;
#include "../Environments/Environment.h"
#include "../Environments/ExternalEnvironment.h"

class ExternalAgent : public Agent
{
public:
    using Agent::Agent;

    State *s;
    Action *a;
    double r;

    ExternalEnvironment* env;

    void   getState(State& s);
    double getReward();
    void   act(Action& a);
    void   move(double dt) {};

    void   setEnvironment(Environment* env);
};


