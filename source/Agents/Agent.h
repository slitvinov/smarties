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
	
	Agent(double newLearningInterval, Types newType, string newName);
	
	virtual void   getState(State& s) { };
	virtual double getReward()        { return 0; }
	virtual void   act(Action& a)     { };
	virtual void   move(double dt) = 0;
	
#ifdef _RL_VIZ
	virtual void   paint()         = 0;
#endif
	
	
	inline StateInfo   getStateDims()   {return sInfo;}
	
	inline ActionInfo  getActionDims()  {return actInfo;}
	
	inline double	   getLastLearned() {return lastLearned;}
	
	inline void 	   setLastLearned(double t) {lastLearned = t;}	
	
	inline double	   getLearningInterval()    {return learningInterval;}
	
	inline string	   getName() {return name;}
	
	inline Types 	   getType() {return type;}
	
	virtual void	   setEnvironment(Environment* env) {environment = env;}
	
	inline void setDims(StateInfo& newSInfo, ActionInfo& newActInfo)
	{
		sInfo.dim = newSInfo.dim;
		sInfo.type = newSInfo.type;
		actInfo.dim = newActInfo.dim;
		
		for (int i=0; i<sInfo.dim; i++)
		{
			sInfo.bounds.push_back     (newSInfo.bounds[i]);
			sInfo.top.push_back        (newSInfo.top[i]);
			sInfo.bottom.push_back     (newSInfo.bottom[i]);
			sInfo.aboveTop.push_back   (newSInfo.aboveTop[i]);
			sInfo.belowBottom.push_back(newSInfo.belowBottom[i]);
		}
		
		for (int i=0; i<actInfo.dim; i++) actInfo.bounds.push_back(newActInfo.bounds[i]);
	}
};





