/*
 *  CouzinEnvironment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 12.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "Environment.h"
#include "CellList.h"

class CouzinAgent;
#include "../Agents/CouzinAgent.h"

class CouzinEnvironment: public virtual Environment
{
public:
	StateInfo sI;
	ActionInfo aI;
	double tm;
	
public:
	vector<CouzinAgent*>   couzins;
	Cells <CouzinAgent>*   cells;
	CellsTraverser<CouzinAgent>* getter;
	
	CouzinEnvironment(vector<Agent*> newAgents);
	
	void findClosestNeighbours(vector<CouzinAgent*>& res, CouzinAgent* agent, double dist);
	
	void evolve(double t);
};