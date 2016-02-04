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
	double lastLearned;
	double learningInterval;
	
	string name;
	static atomic_int idCount;
	
public:
	Types  type;
	int    id;
	
	Agent(double learningInterval, Types type, string name);
	
	virtual void   getState(State& s) { };
	virtual double getReward()        { return 0; }
    virtual double getInfo(int n)     { return 0; }
	virtual void   act(Action& a)     { };
	virtual void   move(double dt) = 0;
	
	inline StateInfo   getStateDims()   {return sInfo;}
	
	inline ActionInfo  getActionDims()  {return actInfo;}
	
	inline double	   getLastLearned() {return lastLearned;}
	
	inline void 	   setLastLearned(double t) {lastLearned = t;}	
	
	inline double	   getLearningInterval()    {return learningInterval;}
	
	inline string	   getName() {return name;}
	
	inline Types 	   getType() {return type;}
	
	virtual void	   setEnvironment(Environment* env) {environment = env;}
	
	inline void setDims(StateInfo& sInfo, ActionInfo& actInfo)
	{
		this->sInfo.dim   = sInfo.dim;
		this->sInfo.type  = sInfo.type;
		this->actInfo.dim = actInfo.dim;
		
		for (int i=0; i<sInfo.dim; i++)
		{
		    this->sInfo.bounds.push_back     (sInfo.bounds[i]);
		    this->sInfo.top.push_back        (sInfo.top[i]);
		    this->sInfo.bottom.push_back     (sInfo.bottom[i]);
		    this->sInfo.aboveTop.push_back   (sInfo.aboveTop[i]);
		    this->sInfo.belowBottom.push_back(sInfo.belowBottom[i]);
		}
		
		for (int i=0; i<actInfo.dim; i++)
		    this->actInfo.bounds.push_back(actInfo.bounds[i]);
	}
};





