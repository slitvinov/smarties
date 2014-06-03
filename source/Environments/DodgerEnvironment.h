/*
 *  DodgerEnvironment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "Environment.h"

class CircularWall;
class DynamicColumn;
class SmartyDodger;
#include "../Agents/SmartyDodger.h"
#include "../Agents/DynamicColumn.h"
#include "../Agents/CircularWall.h"

class DodgerEnvironment: public Environment
{
private:
	void setDims();
	
public:
	vector<DynamicColumn*> dynColumns;
	vector<SmartyDodger*>  dodgers;
	CircularWall*          circWall;
	
	DodgerEnvironment(vector<Agent*> newAgents);
	DynamicColumn* findClosestDynColumn(SmartyDodger* agent);
	
};
