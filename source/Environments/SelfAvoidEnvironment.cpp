/*
 *  SelfAvoidEnvironment.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "SelfAvoidEnvironment.h"
#include "../Misc.h"
#include "../ErrorHandling.h"
#include "../Settings.h"

using namespace ErrorHandling;

SelfAvoidEnvironment::SelfAvoidEnvironment(vector<Agent*> newAgents): Environment(newAgents)
{
	sI.dim = 7;
	sI.bounds.push_back(10);	
	sI.bounds.push_back(10);	
	sI.bounds.push_back(10);	
	sI.bounds.push_back(10);	
	sI.bounds.push_back(10);	
	sI.bounds.push_back(10);	
	sI.bounds.push_back(10);
	
	aI.dim = 1;
	for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
	
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
		if ((*it)->getName() == "SmartySelfAvoider")
		{
			dodgers.push_back(static_cast<SmartySelfAvoider*> (*it));
			(static_cast<SmartySelfAvoider*> (*it))->setDims(sI, aI);
			continue;
		}
		else die("Dodger environment doesn't support objects of type %s\n", (*it)->getName().c_str());
	}
	
	double x0 = circWall->x;
	double y0 = circWall->y;
	double d  = circWall->d;
	cells =  new Cells<SmartySelfAvoider>(dodgers, 8*dodgers[0]->d, x0-d, y0-d, x0+d, y0+d);
	totalReward = 0;
}

DynamicColumn* SelfAvoidEnvironment::findClosestDynColumn(SmartySelfAvoider* agent)
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

SmartySelfAvoider* SelfAvoidEnvironment::findClosestNeighbour(SmartySelfAvoider* agent)
{
	double min = 1e10;	
	double xj, yj;
	
	CellsTraverser<SmartySelfAvoider>* getter = new CellsTraverser<SmartySelfAvoider>(cells);
	getter->prepare(cells->getObjId(agent));
	SmartySelfAvoider *n, *closest = NULL;
	while (getter->getNextXY(xj, yj, n))
	{
		double dst = _dist(agent->x, agent->y, xj, yj);
		if (dst < min)
		{
			min = dst;
			closest = n;
		}
	}
	return closest;
}

void SelfAvoidEnvironment::evolve(double t)
{
	cells->migrate();
}
	
