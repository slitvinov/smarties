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
    vector<Real> max_scale, min_scale;
	map<string, void*> data;
	int bStatus;
    Environment() {};
    Environment(vector<Agent*> agents) : agents(agents), max_scale(20, -1000), min_scale(20, 1000) {};
    
    virtual void setDims () =0;
	virtual void getState (Agent* agent,State& s) { };
	virtual Real getReward(Agent* agent)          { return 0; };
	virtual int  evolve   (Real t)                { return 0; };
    virtual int  init     ()                { return 0; };
    virtual void close_Comm ( ) {};
    virtual void setup_Comm ( ) {};
    virtual bool pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward) {return false;}
};
