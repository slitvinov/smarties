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

#include "../AllSystems.h"

class DodgerEnvironment: public Environment
{
private:
	StateInfo sI;
	ActionInfo aI;
	
public:
	vector<DynamicColumn*> dynColumns;
	CircularWall*          circWall;
	
	DodgerEnvironment(vector<Agent*> newAgents);
	DynamicColumn* findClosestDynColumn(SmartyDodger* agent);
	
};
