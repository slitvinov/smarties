/*
 *  SmartySelfAvoider.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "../StateAction.h"
#include "Agent.h"

class SelfAvoidEnvironment;
#include "../Environments/Environment.h"
#include "../Environments/SelfAvoidEnvironment.h"

class SmartySelfAvoider : public Agent
{
public:
	SelfAvoidEnvironment* env;
	
	bool crashed;	
		
	void _rotate(double dAng);
	
public:
	double x, y, IvI, vx, vy, d;

	SmartySelfAvoider(double x, double y, double d, double t,
				      double vx = 5.0, double vy = 0.0);
		
	void   getState(State& s);
	double getReward();
	void   act(Action& a);
	void   move(double dt);
	
	void   setEnvironment(Environment* env);
};
