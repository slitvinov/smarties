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
			//(static_cast<SmartySelfAvoider*> (*it))->setDims(sI, aI);
			continue;
		}
		else die("Dodger environment doesn't support objects of type %s\n", (*it)->getName().c_str());
	}
					 
	double x0 = circWall->x;
	double y0 = circWall->y;
	double d  = circWall->d;
	cells =  new Cells<SmartySelfAvoider>(dodgers, 8*dodgers[0]->d, x0-d, y0-d, x0+d, y0+d);
	totalReward = 0;
	
	setDims();
	for (vector<SmartySelfAvoider*>::iterator it = dodgers.begin(); it != dodgers.end(); it++)
		(*it)->setDims(sI, aI);

}

void SelfAvoidEnvironment::setDims()
{
	double d = dodgers[0]->d;
	sI.dim = 6;
	sI.type = DISCR;
	
	// dist to wall
	sI.bounds.push_back(10);
	sI.top.push_back(d);
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

	
	// dist to neigh
	sI.bounds.push_back(10);
	sI.top.push_back(d*2);
	sI.bottom.push_back(-d);
	sI.aboveTop.push_back(true);
	sI.belowBottom.push_back(true);
	
	// angle to neigh
	sI.bounds.push_back(10);
	sI.top.push_back(360);
	sI.bottom.push_back(0);
	sI.aboveTop.push_back(false);
	sI.belowBottom.push_back(false);
	
	aI.dim = 1;
	for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
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
	
