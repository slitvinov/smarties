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
    
	vector<Agent*> agents;
	map<string, void*> data;
	
    Environment() {};
    Environment(vector<Agent*> agents) : agents(agents) {};
	
	virtual void   getState (Agent* agent, State& s) { };
	virtual double getReward(Agent* agent)           { return 0; };
	virtual int   evolve   (double t)                { return 0; };
};
