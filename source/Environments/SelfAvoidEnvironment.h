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
#include <tuple>

#include "Environment.h"
#include "CellList.h"
#include "../Agents/SmartySelfAvoider.h"

class SmartySelfAvoider;

typedef tuple<double, double, double> Column;

class SelfAvoidEnvironment: public Environment
{
    void setDims();

public:
	vector<Column> columns;
	vector<SmartySelfAvoider*> avoiders;
	double rWall;
	Cells <SmartySelfAvoider>* cells;

	SelfAvoidEnvironment(vector<Agent*> agents, vector<Column> columns, double rWall, StateType st);
	Column findClosestColumn(SmartySelfAvoider* agent);
	SmartySelfAvoider* findClosestNeighbour(SmartySelfAvoider* agent);
	
	int evolve(double t);
};
