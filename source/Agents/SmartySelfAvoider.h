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
		
	void _rotate(Real dAng);
	
public:
	Real x, y, IvI, vx, vy, d;

	SmartySelfAvoider(Real x, Real y, Real d, Real t,
				      Real vx = 5.0, Real vy = 0.0);
		
	void   getState(State& s);
	Real getReward();
	void   act(Action& a);
	void   move(Real dt);
	
	void   setEnvironment(Environment* env);
};
