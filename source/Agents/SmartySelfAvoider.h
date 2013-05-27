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
#include "../AllSystems.h"

class SmartySelfAvoider : public Agent
{
public:
	SelfAvoidEnvironment* environment;
	DynamicColumn* closestDynCol;
	SmartySelfAvoider* closestNeighbour;
	
	bool crashed;	
		
	void _rotate(double dAng);
	
public:
	double x, y, IvI, vx, vy, d;

	SmartySelfAvoider(double newX, double newY, double newD,  double newT,
				      double newVx = 5.0, double newVy = 0.0);
	
	void   setDims(StateInfo& newSInfo, ActionInfo& newActInfo);
	
	void   getState(State& s);
	double getReward();
	void   act(Action a);
	void   move(double dt);
	
	void   setEnvironment(Environment* env);
	
#ifdef _RL_VIZ
	void   paint();
#endif
};