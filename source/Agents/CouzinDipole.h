/*
 *  CouzinDipole.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 18.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "../StateAction.h"
#include "Agent.h"
#include "FluidAgent.h"
#include "CouzinAgent.h"

class CouzinDipoleEnvironment;
#include "../Environments/Environment.h"

class CouzinDipole : public FluidAgent, public CouzinAgent
{
public:
	CouzinDipoleEnvironment* environment;
	
public:
	double alpha, l;
	
	CouzinDipole(double newX, double newY, double newD,  double newT, double newDomainSize, 
				 double newZor, double newZoo, double newZoa, double newAngle, double newTurnRate, double newVx,  double newVy, RNG* newRng);
	
	void move(double dt);
	void setEnvironment(Environment* env);
#ifdef _RL_VIZ
	void   paint();
#endif
};