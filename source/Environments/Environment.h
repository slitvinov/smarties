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

class Agent;

#include "../Agents/Agent.h"
#include "../StateAction.h"

class Environment
{
public:
	vector<Agent*> agents;
	
	Environment(vector<Agent*> newAgents): agents(newAgents) {};
	
	virtual void   getState (Agent* agent, State& s) { };
	virtual double getReward(Agent* agent)           { return 0; }
	virtual void   evolve   (double t)               { };
};

struct System
{
	Environment* env;
	vector<Agent*> agents;
};