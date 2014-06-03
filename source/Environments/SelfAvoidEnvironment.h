/*
 *  SelfAvoidEnvironment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "Environment.h"
#include "CellList.h"

class CircularWall;
class DynamicColumn;
class SmartySelfAvoider;
#include "../Agents/SmartySelfAvoider.h"
#include "../Agents/DynamicColumn.h"
#include "../Agents/CircularWall.h"

class SelfAvoidEnvironment: public Environment
{
public:
	void setDims();
		
public:
	vector<DynamicColumn*> dynColumns;
	CircularWall*          circWall;
	vector<SmartySelfAvoider*>   dodgers;
	Cells <SmartySelfAvoider>*   cells;

	SelfAvoidEnvironment(vector<Agent*> newAgents, StateType tp);
	DynamicColumn*     findClosestDynColumn(SmartySelfAvoider* agent);
	SmartySelfAvoider* findClosestNeighbour(SmartySelfAvoider* agent);
	
	void evolve(double t);
};