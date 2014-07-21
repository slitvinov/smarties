/*
 *  SwarmEnvironment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 09.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "Environment.h"
#include "CellList.h"

class SmartySwarmer;
#include "../Agents/SmartySwarmer.h"
#include "PotentialFluidEnvironment.h"


class SwarmEnvironment: public PotentialFluidEnvironment
{
public:
	void setDims();
	double momentum;
	void calculateMomentum();	
	
public:
	vector<SmartySwarmer*>   swarmers;
	Cells <SmartySwarmer>*   cells;
	CellsTraverser<SmartySwarmer>* getter;
	
	SwarmEnvironment(vector<Agent*> newAgents, StateType tp);
	
	void findClosestNeighbours(vector<SmartySwarmer*>& res, SmartySwarmer* agent, double dist);
	SmartySwarmer* findClosestNeighbour(SmartySwarmer* agent);	
	void evolve(double t);
	
	double getMomentum()
	{
		return momentum;
	}
};