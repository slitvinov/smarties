/*
 *  SwarmEnvironment.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 09.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <unistd.h>

#include "SwarmEnvironment.h"
#include "../Misc.h"

using namespace ErrorHandling;

SwarmEnvironment::SwarmEnvironment(vector<Agent*> newAgents, StateType tp): PotentialFluidEnvironment(newAgents), Environment(newAgents)
{
	for (vector<Agent*>::iterator it = agents.begin(); it != agents.end(); it++)
	{
		SmartySwarmer* swarmer = dynamic_cast<SmartySwarmer*> (*it);
		if (swarmer != NULL)
		{
			swarmers.push_back(swarmer);
			continue;
		}
		else die("Swarming environment doesn't support objects of type %s\n", (*it)->getName().c_str());
	}
	
	double d = swarmers[0]->domainSize;
	double l = swarmers[0]->zoa;
	if (d / l < 4) die("Zone of attraction is too big for this domain. Consider rescaling\n");
	
	cells =  new Cells<SmartySwarmer>(swarmers, swarmers[0]->zoa, settings.centerX-d/2, settings.centerY-d/2, settings.centerX+d/2, settings.centerY+d/2);
	getter = new CellsTraverser<SmartySwarmer>(cells);
	
	storeDataRef(&swarmers, "swarmers");
	
	sI.type = tp;
	setDims();
	for (vector<SmartySwarmer*>::iterator it = swarmers.begin(); it != swarmers.end(); it++)
		(*it)->setDims(sI, aI);
	
}

void SwarmEnvironment::setDims()
{
	double d = swarmers[0]->d;
	double v = swarmers[0]->IvI;
	
	int nNeigh = 1;
	
	sI.dim = 1 + 2*nNeigh;
	
	for (int i=0; i<nNeigh; i++)
	{
		// dist to neigh
		sI.bounds.push_back(20);
		sI.top.push_back(10*d);
		sI.bottom.push_back(0);
		sI.aboveTop.push_back(true);
		sI.belowBottom.push_back(true);
		
		// angle to neigh
		sI.bounds.push_back(20);
		sI.top.push_back(180);
		sI.bottom.push_back(-180);
		sI.aboveTop.push_back(false);
		sI.belowBottom.push_back(false);
	}
	
	sI.bounds.push_back(20);
	sI.top.push_back(180);
	sI.bottom.push_back(-180);
	sI.aboveTop.push_back(false);
	sI.belowBottom.push_back(false);
	
	aI.dim = 1;
	for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
}


void SwarmEnvironment::findClosestNeighbours(vector<SmartySwarmer*>& res, SmartySwarmer* agent, double dist)
{
	res.clear();
	double xj, yj;
	
	getter->prepare(cells->getObjId(agent));
	SmartySwarmer *n;
	
	while (getter->getNextXY(xj, yj, n))
	{
		double dst = _dist(agent->x, agent->y, xj, yj);
		if (dst < dist && n->type != DEAD)
		{
			n->physX = xj;
			n->physY = yj;
			res.push_back(n);
		}
	}
	//res = swarmers;
}

SmartySwarmer* SwarmEnvironment::findClosestNeighbour(SmartySwarmer* agent)
{
	double min = 1e10;	
	double xj, yj;
	
	getter->prepare(cells->getObjId(agent));
	SmartySwarmer *n, *closest = NULL;
	while (getter->getNextXY(xj, yj, n))
	{
		double dst = _dist(agent->x, agent->y, xj, yj);
		if (dst < min)
		{
			n->physX = xj;
			n->physY = yj;
			min = dst;
			closest = n;
		}
	}
	return closest;
}

void SwarmEnvironment::calculateMomentum()
{
	int tot = 0;
	double resx = 0;
	double resy = 0;
	for (int i=0; i<swarmers.size(); i++)
	{ 
		if (swarmers[i]->getType() != DEAD)
		{
			resx += swarmers[i]->vx;
			resy += swarmers[i]->vy;
			tot++;
		}
	}
	
	momentum = hypot(resx, resy) / tot;
}

void SwarmEnvironment::evolve(double t)
{
	cells->migrate();
	PotentialFluidEnvironment::getVelocities(settings.immortal);
	calculateMomentum();
	//usleep(200);
}
