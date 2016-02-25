/*
 *  Agent.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <string>
#include <atomic>

#include "../StateAction.h"
#include "../Settings.h"
class Environment;
#include "../Environments/Environment.h"

#ifdef _RL_VIZ
#ifdef __APPLE__
#include "GLUT/glut.h"
#endif
#endif

using namespace std;

enum Types {ACTOR, IDLER, DEAD};

class Agent
{
protected:
	Environment* environment;
	
	StateInfo  sInfo;
	ActionInfo actInfo;
	Real lastLearned;
	Real learningInterval;
	
	string name;
	static atomic_int idCount;
	
public:
	Types  type;
	int    id;
	
	Agent(Real learningInterval, Types type, string name);
	
	virtual void   getState(State& s) { };
	virtual Real getReward()        { return 0; }
    virtual Real getInfo(int n)     { return 0; }
	virtual void   act(Action& a)     { };
	virtual void   move(Real dt) = 0;
	
	inline StateInfo   getStateDims()   {return sInfo;}
	
	inline ActionInfo  getActionDims()  {return actInfo;}
	
	inline Real	   getLastLearned() {return lastLearned;}
	
	inline void 	   setLastLearned(Real t) {lastLearned = t;}	
	
	inline Real	   getLearningInterval()    {return learningInterval;}
	
	inline string	   getName() {return name;}
	
	inline Types 	   getType() {return type;}
	
	virtual void	   setEnvironment(Environment* env) {environment = env;}
	
	inline void setDims(StateInfo& stateInfo, ActionInfo& actionInfo)
	{
		this->sInfo.dim   = stateInfo.dim;
		this->sInfo.type  = stateInfo.type;
		this->actInfo.dim = actionInfo.dim;
		
		for (int i=0; i<stateInfo.dim; i++)
		{
		    this->sInfo.bounds.push_back     (stateInfo.bounds[i]);
		    this->sInfo.top.push_back        (stateInfo.top[i]);
		    this->sInfo.bottom.push_back     (stateInfo.bottom[i]);
		    this->sInfo.aboveTop.push_back   (stateInfo.aboveTop[i]);
		    this->sInfo.belowBottom.push_back(stateInfo.belowBottom[i]);
            this->sInfo.isLabel.push_back    (stateInfo.isLabel[i]);
		}
        
		for (int i=0; i<actionInfo.dim; i++)
        {
		    this->actInfo.bounds.push_back(actionInfo.bounds[i]);
        }
        
        for (int i=0; i<actionInfo.values.size(); i++)
        {
            this->actInfo.values.push_back(actionInfo.values[i]);
        }
        for (int i=0; i<stateInfo.values.size(); i++)
        {
            this->sInfo.values.push_back(stateInfo.values[i]);
        }
	}
};
