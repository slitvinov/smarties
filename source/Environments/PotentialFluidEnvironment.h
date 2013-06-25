/*
 *  PotentialFluidEnvironment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 18.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "CellList.h"

class FluidAgent;
#include "../Agents/FluidAgent.h"

class PotentialFluidEnvironment : public virtual Environment
{	
public:
	double minDist;
	
	vector<FluidAgent*> fagents;
	vector<FluidAgent*> myfAgents;
	vector<double> vortices;
	vector<pair<double, double> > vortCoos;
	vector<pair<double, double>* > targets;
	
	PotentialFluidEnvironment(vector<Agent*> newAgents);
	
	void getVelocities();
};