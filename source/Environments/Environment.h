/*
 *  Environment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <map>
#include <string>

class Agent;

#include "../Agents/Agent.h"
#include "../StateAction.h"

class Environment
{
public:
    StateInfo sI;
	ActionInfo aI;
        int nInfo;
	vector<Agent*> agents;
	map<string, void*> data;
	int bStatus;
    Environment() {};
    Environment(vector<Agent*> agents) : agents(agents) {};
    
    virtual void setDims () =0;
	virtual void getState (Agent* agent,State& s) { };
	virtual Real getReward(Agent* agent)          { return 0; };
	virtual int  evolve   (Real t)                { return 0; };
    virtual int  init     ()                { return 0; };
    virtual void close_Comm ( ) {};
    virtual void setup_Comm ( ) {};
};
