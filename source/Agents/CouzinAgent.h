/*
 *  CouzinAgent.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 12.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "../StateAction.h"
#include "../rng.h"
#include "Agent.h"

class CouzinEnvironment;
#include "../Environments/Environment.h"
#include "../Environments/CouzinEnvironment.h"

class CouzinAgent : public virtual Agent
{
public:
	CouzinEnvironment* environment;
	
	vector<CouzinAgent*> guys;
	
	void _rotate(double dAng);
	bool _isVisible(double x0, double y0);
	RNG* rng;
	
public:
	double x, y, IvI, vx, vy, d;
	double physX, physY;
	double dx, dy;
	double domainSize;
	
	double zor, zoo, zoa, angle, turnRate;
	
	CouzinAgent(double newX, double newY, double newD,  double newT, double newDomainSize,
				double newZor, double newZoo, double newZoa, double newAngle, double newTurnRate,
				double newVx, double newVy, RNG* newRng);
	
	void   act();
	virtual void move(double dt);
	
	void   setEnvironment(Environment* env);
	
#ifdef _RL_VIZ
	void   paint();
#endif
};