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
}

void PotentialFluidEnvironment::getVelocities()
{
	// Prepare vortices - pack strengths, coordinates and desired velocities into vectors
	int k=0;
	for (int n = 0; n < (int) fagents.size(); n++)
	{
		FluidAgent* agent = fagents[n];
		
		for (int j = 0; j < (int) agent->vortices.size(); j++)
		{
			vortices[k] = agent->vortices[j];
			vortCoos[k].first  = agent->vortCoos[j].first;
			vortCoos[k].second = agent->vortCoos[j].second;
			targets[k]  = &(agent->vortVels[j]);
			k++;
		}
	}
	
	// Solve n^2 problem for all vortices
	int tot = vortices.size();
	for (int n = 0; n < (int) targets.size(); n++)
	{
		complex<double> velocity(0, 0);
		
		double xn = vortCoos[n].first;
		double yn = vortCoos[n].second;
		
		for (int j = 0; j < tot; j++)
		{
			double xj = vortCoos[j].first;
			double yj = vortCoos[j].second;
			double gammaj = vortices[j];
			double X = xn - xj;
			double Y = yn - yj;
			velocity += (j == n) ? (double) (0) : -gammaj * (complex<double>(Y, X)) / (double) (X * X + Y * Y);
		}
		*(targets[n]) = pair<double, double> (real(velocity), imag(velocity)); /// beware, these are conjugate velocities
	}
}






