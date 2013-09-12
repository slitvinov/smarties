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

SmartySwarmer::SmartySwarmer(double newX, double newY, double newD,  double newT, double newDomainSize,
							 double newZoo, double newZoa, double newIvI,  double newAlpha, RNG* newRng):
FluidAgent (ACTOR, "SmartySwarmer", newT), Agent (newT, ACTOR, "SmartySwarmer"),
alpha(newAlpha), IvI(newIvI), d(newD), x(newX), y(newY), domainSize(newDomainSize), rng(newRng), zoo(newZoo), zoa(newZoa)
{
	const double SQRT2PI = sqrt(2*M_PI);
	l = d / (2*SQRT2PI);
	gamma = 2 * M_PI * l * IvI;
	
	double rho = 10 * l;
	double speedK = 0.1;
	gammaA = gamma * speedK;
	gammaT = SQRT2PI * l / rho * gamma;
	
	vortices.resize(2);
	vortCoos.resize(2);
	vortVels.resize(2);
	
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
	
	movState = NORMAL;
};

void SmartySwarmer::setEnvironment(Environment* env)
{
	environment = dynamic_cast<SwarmEnvironment*>(env);
	Agent::environment = env;
}

void SmartySwarmer::computeVecs()
{
	vector<SmartySwarmer*> guys;
	// 2. Zone of orientation
	environment->findClosestNeighbours(guys, this, zoo);
	
	double dxo = 0;
	double dyo = 0;
	for (int i=0; i<guys.size(); i++)
		//if (_isVisible(guys[i]->physX , guys[i]->physY))
		{
			dxo += guys[i]->vx/guys[i]->IvI;
			dyo += guys[i]->vy/guys[i]->IvI;
		}
	
	// 3. Zone of attraction
	environment->findClosestNeighbours(guys, this, zoa);
	
	double dxa = 0;
	double dya = 0;
	for (int i=0; i<guys.size(); i++)
		if (//_isVisible(guys[i]->physX , guys[i]->physY) &&
			_dist(x, y, guys[i]->physX, guys[i]->physY) > zoo)
		{
			double rx = guys[i]->physX - x;
			double ry = guys[i]->physY - y;
			double IrI = hypot(rx, ry);
			
			dxa += rx/IrI;
			dya += ry/IrI;
		}
	
	double dxs = dxo/2 + dxa/2;
	double dys = dyo/2 + dya/2;
	double IdI = hypot(dxs, dys);
	if (IdI > 1e-7)
	{
		dx = dxs;
		dy = dys;
	}
	else 
	{
		dx = vx;
		dy = vy;
	}
	
	//dx = 0.1;
	//dy = -0.2;
}

void SmartySwarmer::move(double dt)
{
	if (type == DEAD)
	{
		vortices[0] = 0;
		vortices[1] = 0;
		return;
	}
	
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
	
	if (x < settings.centerX - domainSize/2) x += domainSize;
	if (x > settings.centerX + domainSize/2) x -= domainSize;
	if (y < settings.centerY - domainSize/2) y += domainSize;
	if (y > settings.centerY + domainSize/2) y -= domainSize;
	
	closestNeighbour = environment->findClosestNeighbour(this);
}

void SmartySwarmer::getState(State& s)
{	
	s.vals.clear();
	//computeVecs();

	vector<SmartySwarmer*> guys;
	environment->findClosestNeighbours(guys, this, sInfo.top[0]);

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
	
	s.vals.push_back( _angle(dx, dy, vx, vy) );
}

double SmartySwarmer::getReward()
{
	double reward = 0;
		
//	SmartySwarmer* closestNeighbour = environment->findClosestNeighbour(this);
//	if ( closestNeighbour != NULL && _dist(x, y, closestNeighbour->physX, closestNeighbour->physY) - 1.4*d < 0 )
//	{
//		//reward -= 1;
//		alpha = -alpha;
//	}
//	
//	if (movState == TURN || movState == FAST) reward -= 0.5;
//	//if (movState == SLOW) reward += 1;
//	
//	double vreward = abs(environment->getMomentum() - IvI)/IvI;
//	reward += - vreward;
//	
//	environment->accumulateReward(reward);
//	return reward;
	
	if ( closestNeighbour != NULL )
	{
		double dst = _dist(x, y, closestNeighbour->physX, closestNeighbour->physY);
		
		if (dst < 1.0*d)
			reward += -50.0 * (d - dst);
		
		if (dst < 0.5*d)
		{
			if (settings.immortal)
				alpha = atan2(y - closestNeighbour->physY, x - closestNeighbour->physX);
		}
			
	}		
	
	if (movState == TURN) reward -= 0.005;
	
	//computeVecs();
	double desiredAng = abs(_angle(dx, dy, vx, vy) / 180);
	//reward += -0.5 * pow(desiredAng, 1.0);
	//if (desiredAng * 180 > 10) reward -= 0.5; 
	
	environment->accumulateReward(reward);
	return reward;
}


void SmartySwarmer::act(Action a)
{
	vortices[0] = gamma;
	vortices[1] = -gamma;
	
	switch (a.vals[0])
	{
		case 0:
			// Move on
			movState = NORMAL;
			break;
			
		case 1:
			// Turn left
			vortices[0] += gammaT;
			vortices[1] += gammaT;
			movState = TURN;
			break;
			
		case 2:
			// Turn right
			vortices[0] -= gammaT;
			vortices[1] -= gammaT;
			movState = TURN;
			break;
			
		case 3:
			// Speed up
			vortices[0] += gammaA;
			vortices[1] -= gammaA;
			movState = FAST;
			break;
			
		case 4:
			// Slow down
			vortices[0] -= gammaA;
			vortices[1] += gammaA;
			movState = SLOW;
			break;
			
		default:
			die("aaaaaAAAAAAAAAAAAAAAAAAA!!!!!!!!!!!!");
			break;
	}
}

#ifdef _RL_VIZ
void SmartySwarmer::paint()
{
	//_drawSphere(d/2, x, y, 0, 1, 0);
	if (type != DEAD) _drawArrow (d,   x, y, IvI * cos(alpha), IvI * sin(alpha), IvI, 0, 1, 0);
	else			  _drawSphere(d/3, x, y, 1, 0, 0);
}
#endif


