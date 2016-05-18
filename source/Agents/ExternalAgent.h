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

//class ExternalEnvironment;
//#include "../Environments/Environment.h"
//#include "../Environments/ExternalEnvironment.h"

class ExternalAgent : public Agent
{
public:
    using Agent::Agent;

    State *s;
    Action *a;
    Real r;
    vector<Real> Info;
    
    //ExternalEnvironment* env;

    void   getState(State& s);
    
    Real getReward();
    Real getInfo(int n);
    
    void   act(Action& a);
    void   move(Real dt) {};

    //void   setEnvironment(Environment* env);
};


