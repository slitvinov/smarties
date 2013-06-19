/*
 *  FluidAgent.h
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
#include "../Environments/Environment.h"

class PotentialFluidEnvironment;
#include "../Environments/PotentialFluidEnvironment.h"

class FluidAgent : public virtual Agent
{
public:
	PotentialFluidEnvironment* environment;
		
public:
	vector<pair<double, double> > vortCoos;
	vector<pair<double, double> > vortVels;
	vector<double> vortices;
	
	FluidAgent(Types type, string name, double newT) : Agent(newT, type, name) {};
};