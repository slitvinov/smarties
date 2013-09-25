/*
 *  PotentialFluidEnvironment.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 18.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <complex>

#include "PotentialFluidEnvironment.h"
#include "../Misc.h"

const complex <double> I(0,1);

PotentialFluidEnvironment::PotentialFluidEnvironment(vector<Agent*> agents) : Environment(agents)
{
	int n = 0;
	for (vector<Agent*>::iterator it = agents.begin(); it != agents.end(); it++)
	{
		FluidAgent* fagent = dynamic_cast<FluidAgent*> (*it);
		if (fagent != NULL)
		{
			fagents.push_back(fagent);
			n += fagent->vortices.size();
			continue;
		}
		else die("Potential fluid environment doesn't support objects of type %s\n", (*it)->getName().c_str());
	}
	
	vortices.resize(n);
	vortCoos.resize(n);
	targets.resize(n);
	myfAgents.resize(n);
	
	storeDataRef(&fagents, "fagents");
}

void PotentialFluidEnvironment::getVelocities(bool immortal)
{
	// Prepare vortices - pack strengths, coordinates and desired velocities into vectors
	int k=0;
	double minDist = 1e10;
	for (int n = 0; n < (int) fagents.size(); n++)
	{
		FluidAgent* agent = fagents[n];
		
		for (int j = 0; j < (int) agent->vortices.size(); j++)
		{
			vortices[k] = agent->vortices[j];
			vortCoos[k].first  = agent->vortCoos[j].first;
			vortCoos[k].second = agent->vortCoos[j].second;
			targets[k]  = &(agent->vortVels[j]);
			myfAgents[k] = agent;
			k++;
		}
		
		for (int i = 1; i <= (int) agent->vortices.size(); i++)
			for (int j = i+1; j <= (int) agent->vortices.size(); j++)
			{
				double dist = _dist(vortCoos[k-i].first, vortCoos[k-i].second, vortCoos[k-j].first, vortCoos[k-j].second);
				if (minDist > dist) minDist = dist;
			}
	}
	
	// Solve n^2 problem for all vortices
	int tot = vortices.size();
	for (int n = 0; n < (int) targets.size(); n++)
	{
		if (myfAgents[n]->type == DEAD) continue;
		complex<double> velocity(0, 0);
		
		double xn = vortCoos[n].first;
		double yn = vortCoos[n].second;
		
		for (int j = 0; j < tot; j++)
		{
			if (n == j) continue;
			if (myfAgents[j]->type == DEAD) continue;
			
			double xj = vortCoos[j].first;
			double yj = vortCoos[j].second;
			double gammaj = vortices[j];
			double X = xn - xj;
			double Y = yn - yj;
			
			double dist2 = (X * X + Y * Y);
			//if (!immortal && dist2 < minDist * minDist / 4) myfAgents[n]->type = myfAgents[j]->type = DEAD;
			
			velocity += -gammaj * (complex<double>(Y, X)) / dist2 ;
		}
		
		velocity /= 2 * M_PI;
		
		*(targets[n]) = pair<double, double> (real(velocity), imag(velocity)); /// beware, these are conjugate velocities
	}
}






