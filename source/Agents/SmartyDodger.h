/*
 *  SmartyDodger.h
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
#include "DynamicColumn.h"

class DodgerEnvironment;
#include "../Environments/Environment.h"
#include "../Environments/DodgerEnvironment.h"

class SmartyDodger : public Agent
{
private:
	DodgerEnvironment* environment;
	DynamicColumn* closestDynCol;
	bool crashed;
	
public:
	double x, y, IvI, vx, vy, d;
		
	void _rotate(double dAng);

	SmartyDodger(double newX, double newY, double newD,  double newT,
				 double newVx = 5.0, double newVy = 0.0);
			
	void   getState(State& s);
	double getReward();
	void   act(Action& a);
	void   move(double dt);
	
	void   setEnvironment(Environment* env);
	
#ifdef _RL_VIZ
	void   paint();
#endif
};