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

#include "../StateAction.h"
#include "../Environments/Environment.h"

#ifdef _RL_VIZ
#ifdef __APPLE__
#include "GLUT/glut.h"
#endif
#endif

using namespace std;

enum Types {ACTOR, IDLER};

class Agent
{
protected:
	Environment* environment;
	
	Types  type;
	
	StateInfo  sInfo;
	ActionInfo actInfo;
	double lastLearned;
	double learningInterval;
	
	string name;
	
public:
	Agent(double newLearningInterval, Types newType, string newName):
	      learningInterval(newLearningInterval), lastLearned(0), type(newType), name(newName) { };
	
	virtual void   getState(State& s) { };
	virtual double getReward()        { return 0; }
	virtual void   act(Action a)      { };
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
};





