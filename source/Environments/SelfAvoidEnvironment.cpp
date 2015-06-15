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

SelfAvoidEnvironment::SelfAvoidEnvironment(vector<Agent*> agents, vector<Column> columns, double rWall, StateType tp) :
        Environment(agents), columns(columns), rWall(rWall)
{
    for (auto a : agents)
        avoiders.push_back(static_cast<SmartySelfAvoider*>(a));

	cells =  new Cells<SmartySelfAvoider>(avoiders, 10*avoiders[0]->d, -rWall, -rWall, rWall, rWall);
	
	sI.type = tp;
	setDims();
	for (auto& a : avoiders)
	{
	    a->setEnvironment(this);
		a->setDims(sI, aI);
	}
}

void SelfAvoidEnvironment::setDims()
{
	double d = avoiders[0]->d;
	sI.dim = 6;
	
	// dist to wall
	sI.bounds.push_back(10);
	sI.top.push_back(8*d);
	sI.bottom.push_back(-d);
	sI.aboveTop.push_back(true);
	sI.belowBottom.push_back(true);
	
	// angle to wall
	sI.bounds.push_back(10);
	sI.top.push_back(180);
	sI.bottom.push_back(-180);
	sI.aboveTop.push_back(false);
	sI.belowBottom.push_back(false);
	
	
	// dist to column
	sI.bounds.push_back(10);
	sI.top.push_back(8*d);
	sI.bottom.push_back(-d);
	sI.aboveTop.push_back(true);
	sI.belowBottom.push_back(true);
	
	// angle to column
	sI.bounds.push_back(10);
	sI.top.push_back(180);
	sI.bottom.push_back(-180);
	sI.aboveTop.push_back(false);
	sI.belowBottom.push_back(false);

	
	// dist to neigh
	sI.bounds.push_back(10);
	sI.top.push_back(d*8);
	sI.bottom.push_back(-d);
	sI.aboveTop.push_back(true);
	sI.belowBottom.push_back(true);
	
	// angle to neigh
	sI.bounds.push_back(10);
	sI.top.push_back(180);
	sI.bottom.push_back(-180);
	sI.aboveTop.push_back(false);
	sI.belowBottom.push_back(false);
	
	aI.dim = 1;
	for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
}

Column SelfAvoidEnvironment::findClosestColumn(SmartySelfAvoider* agent)
{
	double min = 1e10;
	Column res = make_tuple(1e10, 1e10, 0);

	for (auto c : columns)
	{
		double l = _dist(agent->x, agent->y, get<0>(c), get<1>(c));
		if (l - agent->d/2 - get<2>(c)/2 < min)
		{
			min = l - agent->d/2 - get<2>(c);
			res = c;
		}
	}
	return res;
}

SmartySelfAvoider* SelfAvoidEnvironment::findClosestNeighbour(SmartySelfAvoider* agent)
{
	double min = 1e10;	
	double xj, yj;
	
	static CellsTraverser<SmartySelfAvoider>* getter(new CellsTraverser<SmartySelfAvoider>(cells));
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

int SelfAvoidEnvironment::evolve(double t)
{
	cells->migrate();
    return 0;
}
	
