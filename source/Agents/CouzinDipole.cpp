/*
 *  CouzinDipole.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 18.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <complex>

#include "CouzinDipole.h"
#include "../ErrorHandling.h"
#include "../Misc.h"

const complex<double> I(0, 1);

using namespace ErrorHandling;

CouzinDipole::CouzinDipole(double newX, double newY, double newD,  double newT, double newDomainSize,
						   double newZor, double newZoo, double newZoa, double newAngle, double newTurnRate, double newVx,  double newVy, RNG* newRng):
CouzinAgent(newX, newY, newD, newT, newDomainSize, newZor, newZoo, newZoa, newAngle, newTurnRate, newVx,  newVy, newRng),
FluidAgent (IDLER, "CouzinDipole", newT), Agent (newT, IDLER, "CouzinDipole"), alpha(M_PI/2), l(0)
{
	l = d / (2*sqrt(2*M_PI));
	double gamma = 2 * M_PI * l * IvI;
	
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
};

void CouzinDipole::setEnvironment(Environment* env)
{
	environment = (CouzinDipoleEnvironment*)env;
	Agent::environment = env;
	CouzinAgent::environment = dynamic_cast<CouzinEnvironment*> (env);
}

void CouzinDipole::move(double dt)
{
	if (type == DEAD)
	{
		vortices[0] = 0;
		vortices[1] = 0;
		alpha = -100;
		return;
	}
	
	complex <double> velocityLeft (vortVels[0].first, vortVels[0].second);
	complex <double> velocityRight(vortVels[1].first, vortVels[1].second);
	
	complex <double> conjVelocity = (double)(0.5)*(velocityRight+velocityLeft);
	double alphaDot0 = real((velocityRight-velocityLeft)*exp(I*alpha))/l;
	
	complex <double> vel = conj(conjVelocity);
	vx = real(vel);
	vy = imag(vel);
	
	CouzinAgent::act();
	double sigma = rng->normal(0, 0.5);	
	
	double desiredAng = _angle(dx, dy, vx, vy);
	if (desiredAng > 180.0) desiredAng -= 360.0;
	if (turnRate * dt < fabs(desiredAng))
		desiredAng = copysign(turnRate*dt, desiredAng);
	
	desiredAng += sigma;
	
	double alphaDotDesired = (desiredAng / dt) / 180 * M_PI;
	
		
	x += vx*dt;
	y += vy*dt;
	alpha += alphaDotDesired*dt;
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
	
	double gammaAdd = M_PI * l*l * (alphaDotDesired - alphaDot0);
	vortices[0] += gammaAdd;
	vortices[1] += gammaAdd;
	
	//vx = IvI * cos(alpha);
	//vy = IvI * sin(alpha);
}

#ifdef _RL_VIZ
void CouzinDipole::paint()
{
	if (type != DEAD) _drawArrow (d,   x, y, IvI * cos(alpha), IvI * sin(alpha), IvI, 0, 1, 0);
	else			  _drawSphere(d/3, x, y, 1, 0, 0);
}
#endif


