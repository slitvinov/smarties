/*
 *  DodgerEnvironment.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "DodgerEnvironment.h"
#include "../Misc.h"
#include "../ErrorHandling.h"
#include "../Settings.h"
#include "../StateAction.h"

using namespace ErrorHandling;

DodgerEnvironment::DodgerEnvironment(vector<Agent*> newAgents): Environment(newAgents)
{	
	for (vector<Agent*>::iterator it = agents.begin(); it != agents.end(); it++)
	{
		if ((*it)->getName() == "CircularWall") 
		{
			circWall = static_cast<CircularWall*> (*it);
			continue;
		}
		if ((*it)->getName() == "DynamicColumn")
		{
			dynColumns.push_back(static_cast<DynamicColumn*> (*it));
			continue;
		}
		if ((*it)->getName() == "SmartyDodger")
		{
			dodgers.push_back(static_cast<SmartyDodger*> (*it));// ->setDims(sI, aI);
			continue;
		}
		else die("Dodger environment doesn't support objects of type %s\n", (*it)->getName().c_str());
	}
	
	setDims();
}

void DodgerEnvironment::setDims()
{
	double d = dodgers[0]->d;
	sI.dim = 4;
	sI.type = DISCR;
	
	// dist to wall
	sI.bounds.push_back(10);
	sI.top.push_back(10*d);
	sI.bottom.push_back(0);
	sI.aboveTop.push_back(true);
	sI.belowBottom.push_back(true);
	
	// angle to wall
	sI.bounds.push_back(10);
	sI.top.push_back(360);
	sI.bottom.push_back(0);
	sI.aboveTop.push_back(false);
	sI.belowBottom.push_back(false);
	
	
	// dist to column
	sI.bounds.push_back(10);
	sI.top.push_back(5*d);
	sI.bottom.push_back(0);
	sI.aboveTop.push_back(true);
	sI.belowBottom.push_back(true);
	
	// angle to column
	sI.bounds.push_back(10);
	sI.top.push_back(360);
	sI.bottom.push_back(0);
	sI.aboveTop.push_back(false);
	sI.belowBottom.push_back(false);
	
	aI.dim = 1;
	for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
}


DynamicColumn* DodgerEnvironment::findClosestDynColumn(SmartyDodger* agent)
{
	double min = 1e10;
	DynamicColumn* res = NULL;

	for (vector<DynamicColumn*>::iterator it = dynColumns.begin(); it != dynColumns.end(); it++)
	{
		double l = _dist(agent->x, agent->y, (*it)->x, (*it)->y);
		if (l - agent->d/2 - (*it)->d/2 < min)
		{
			min = l - agent->d/2 - (*it)->d/2;
			res = *it;
		}
	}
	return res;
}


