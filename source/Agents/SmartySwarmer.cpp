/*
 *  SmartySwarmer.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 09.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */


#include <complex>
#include <algorithm>

#include "SmartySwarmer.h"
#include "../ErrorHandling.h"
#include "../Misc.h"

const complex<double> I(0, 1);

using namespace ErrorHandling;

SmartySwarmer::SmartySwarmer(double x, double y, double d,  double T, double domainSize,
							 double IvI,  double alpha):
Agent (T, ACTOR, "SmartySwarmer"),
alpha(alpha), IvI(IvI), d(d), x(x), y(y), domainSize(domainSize)
{
	const double SQRT2PI = sqrt(2*M_PI);
	l = d / (2*SQRT2PI);
	gamma = 2 * M_PI * l * IvI;

	double rho = 10 * l;
	double speedK = 0.1;
	gammaA = gamma * speedK;
	gammaT = SQRT2PI * l / rho * gamma;

	vortices[0] = gamma;
	vortices[1] = -gamma;

	complex<double> locationCenter(x, y);
	complex<double> tmp = I*l*exp(complex<double>(I*alpha))/complex<double>(2,0);
	complex<double> locationRightVortex = locationCenter + tmp;
	complex<double> locationLeftVortex  = locationCenter - tmp;
	vortCoos[0] = pair<double, double>(real(locationRightVortex), imag(locationRightVortex));
	vortCoos[1] = pair<double, double>(real(locationLeftVortex),  imag(locationLeftVortex));

	vortVels[0] = pair<double, double>(0, 0);
	vortVels[1] = pair<double, double>(0, 0);

	vx = IvI*cos(alpha);
	vy = IvI*sin(alpha);
};


void SmartySwarmer::setEnvironment(Environment* env)
{
	this->env = dynamic_cast<SwarmEnvironment*>(env);
	Agent::environment = env;
}

void SmartySwarmer::move(double dt)
{
	complex <double> velocityLeft (vortVels[0].first, vortVels[0].second);
	complex <double> velocityRight(vortVels[1].first, vortVels[1].second);

	complex <double> conjVelocity = (double)(0.5)*(velocityRight+velocityLeft);
	double alphaDot = real((velocityRight-velocityLeft)*exp(I*alpha))/l;

	complex <double> vel = conj(conjVelocity);
	vx = real(vel);
	vy = imag(vel);

	x += vx*dt;
	y += vy*dt;

	alpha += alphaDot*dt;
	alpha = (alpha > M_PI)  ? alpha - 2 * M_PI : alpha;
	alpha = (alpha < -M_PI) ? alpha + 2 * M_PI : alpha;

	complex<double> locationCenter(x, y);
	complex<double> tmp = I*l*exp(complex<double>(I*alpha))/complex<double>(2,0);
	complex<double> locationRightVortex = locationCenter + tmp;
	complex<double> locationLeftVortex  = locationCenter - tmp;
	vortCoos[0] = pair<double, double>(real(locationRightVortex), imag(locationRightVortex));
	vortCoos[1] = pair<double, double>(real(locationLeftVortex),  imag(locationLeftVortex));

	if (x < -domainSize/2) x += domainSize;
	if (x > domainSize/2)  x -= domainSize;
	if (y < -domainSize/2) y += domainSize;
	if (y > domainSize/2)  y -= domainSize;
}

void SmartySwarmer::getState(State& s)
{
	s.vals.clear();

	vector<SmartySwarmer*> guys;
	env->findClosestNeighbours(guys, this, sInfo.top[0]);

	Comparator comp(x, y);
	sort(guys.begin(), guys.end(), comp);

	int maxnum = (sInfo.dim-1) / 2;

	for (int i=0; i < maxnum; i++)
	{
		if (i < guys.size())
		{
			SmartySwarmer* n = guys[i];
			if (n == this)
			{
				maxnum++;
				continue;
			}
			s.vals.push_back(_dist(x, y, n->physX, n->physY));
			s.vals.push_back(_angle(vx - n->vx, vy - n->vy,  -(x - n->physX),  -(y - n->physY)));
			//s.vals.push_back(_angle(vx, vy, n->vx, n->vy));
		}
		else
		{
			s.vals.push_back(sInfo.top[i*2]);
			s.vals.push_back(sInfo.top[i*2+1]);
			//s.vals.push_back(0);
		}

	}
}

double SmartySwarmer::getReward()
{
	double reward = 0;

	SmartySwarmer* closestNeighbour = env->findClosestNeighbour(this);
	if ( closestNeighbour != NULL )
	{
		double dst = _dist(x, y, closestNeighbour->physX, closestNeighbour->physY);

		if (dst < 1.0*d)
			reward += -1.0;
	}

	return reward;
}


void SmartySwarmer::act(Action& a)
{
	vortices[0] = gamma;
	vortices[1] = -gamma;

	switch (a.vals[0])
	{
		case 0:
			// Move on
			break;

		case 1:
			// Turn left
			vortices[0] += gammaT;
			vortices[1] += gammaT;
			break;

		case 2:
			// Turn right
			vortices[0] -= gammaT;
			vortices[1] -= gammaT;
			break;

		case 3:
			// Speed up
			vortices[0] += gammaA;
			vortices[1] -= gammaA;
			break;

		case 4:
			// Slow down
			vortices[0] -= gammaA;
			vortices[1] += gammaA;
			break;

		default:
			die("aaaaaAAAAAAAAAAAAAAAAAAA!!!!!!!!!!!!");
			break;
	}
}


