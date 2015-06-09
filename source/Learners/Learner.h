/*
 *  Learner.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 15.07.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <string>
#include <list>

#include "../Agents/Agent.h"
#include "../Environments/Environment.h"
#include "Trace.h"

using namespace std;

class Learner
{
public:
	Learner() { };
	    
    //virtual void selectAction(State& s, Action& a) = 0;
    //virtual void update(State& s, Action& a, double r, State& s1) = 0;
    virtual void updateSelect(Trace& t, State& s, Action& a, double r) = 0;
    
    virtual void savePolicy(string prefix) = 0;
	virtual void try2restart(string fname) = 0;
};
