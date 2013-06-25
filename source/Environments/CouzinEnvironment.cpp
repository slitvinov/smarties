/*
 *  CouzinEnvironment.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 12.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "CouzinEnvironment.h"
#include "../Misc.h"
#include "../ErrorHandling.h"
#include "../Settings.h"

#include <unistd.h>

using namespace ErrorHandling;

CouzinEnvironment::CouzinEnvironment(vector<Agent*> newAgents): Environment(newAgents)
{
	for (vector<Agent*>::iterator it = agents.begin(); it != agents.end(); it++)
	{
		CouzinAgent* couzin = dynamic_cast<CouzinAgent*> (*it);
		if (couzin != NULL)
		{
			couzins.push_back(couzin);
			continue;
		}
		else die("Couzin environment doesn't support objects of type %s\n", (*it)->getName().c_str());
	}
	
	double d = couzins[0]->domainSize;
	double l = couzins[0]->zoa;
	if (d / l < 4) die("Zone of attraction is too big for this domain. Consider rescaling\n");
	
	cells =  new Cells<CouzinAgent>(couzins, couzins[0]->zoa, settings.centerX-d/2, settings.centerY-d/2, settings.centerX+d/2, settings.centerY+d/2);
	getter = new CellsTraverser<CouzinAgent>(cells);
}

void CouzinEnvironment::findClosestNeighbours(vector<CouzinAgent*>& res, CouzinAgent* agent, double dist)
{
	res.clear();
	double xj, yj;
		
	getter->prepare(cells->getObjId(agent));
	CouzinAgent *n;
	
	while (getter->getNextXY(xj, yj, n))
	{
		n->physX = xj;
		n->physY = yj;
		double dst = _dist(agent->x, agent->y, xj, yj);
		if (dst < dist && n->type != DEAD)
		{
			res.push_back(n);
		}
	}
}

void CouzinEnvironment::evolve(double t)
{
	cells->migrate();
	usleep(000);
}

