/*
 *  CouzinAgent.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 12.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "CouzinAgent.h"
#include "../ErrorHandling.h"
#include "../Misc.h"

using namespace ErrorHandling;

CouzinAgent::CouzinAgent(double newX, double newY, double newD,  double newT, double newDomainSize,
						 double newZor, double newZoo, double newZoa, double newAngle, double newTurnRate, double newVx,  double newVy, RNG* newRng):
Agent(newT, IDLER, "CouzinAgent"), x(newX), y(newY), d(newD), domainSize(newDomainSize),
zor(newZor), zoo(newZoo), zoa(newZoa), angle(newAngle), turnRate(newTurnRate), vx(newVx), vy(newVy)
{
	rng = new RNG(rand());
	IvI = sqrt(vx*vx + vy*vy);
	//double ang = rng.uniform(0, 2*M_PI);
//	
//	vx = IvI * cos(ang);
//	vy = IvI * sin(ang);
	
	sInfo.dim = 0;
	actInfo.dim = 0;
}

void CouzinAgent::setEnvironment(Environment* env)
{
	environment = dynamic_cast<CouzinEnvironment*> (env);
}

void CouzinAgent::_rotate(double dAng)
{
	double ang = _angle(vx, vy, 1, 0) + dAng;
	
	vx = IvI * cos(2*M_PI * ang / 360.0);
	vy = IvI * sin(2*M_PI * ang / 360.0);
}

bool CouzinAgent::_isVisible(double x0, double y0)
{
	double ang = _angle(vx, vy, x0-x, y0-y);
	if (ang > 180.0) ang -= 360.0;
	
	if (fabs(ang) > angle / 2) return false;
	else return true;
}

void CouzinAgent::act()
{
	double IdI;
	
	// 0. Check for collisions
	
	static const double l = d / (2*sqrt(2*M_PI));
	environment->findClosestNeighbours(guys, this, 2 * l);
	//if (guys.size() > 0) type = DEAD;
	
	// 1. Zone of repulsion
	environment->findClosestNeighbours(guys, this, zor);
	
	double dxr = 0;
	double dyr = 0;
	for (int i=0; i<guys.size(); i++)
	{
		double rx = guys[i]->physX - x;
		double ry = guys[i]->physY - y;
		double IrI = hypot(rx, ry);
		
		dxr -= rx/IrI;
		dyr -= ry/IrI;
	}
	IdI = hypot(dxr, dyr);
	
	if (guys.size() > 0)
	{
		dx = dxr;
		dy = dyr;
		return;
	}
	
	// 2. Zone of orientation
	environment->findClosestNeighbours(guys, this, zoo);
	
	double dxo = 0;
	double dyo = 0;
	for (int i=0; i<guys.size(); i++)
		if (_isVisible(guys[i]->physX , guys[i]->physY))
		{
			dxo += guys[i]->vx/guys[i]->IvI;
			dyo += guys[i]->vy/guys[i]->IvI;
		}

	// 3. Zone of attraction
	environment->findClosestNeighbours(guys, this, zoa);
	
	double dxa = 0;
	double dya = 0;
	for (int i=0; i<guys.size(); i++)
		if (_isVisible(guys[i]->physX , guys[i]->physY) &&
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
	IdI = hypot(dxs, dys);
	if (IdI > 1e-8)
	{
		dx = dxs;
		dy = dys;
	}
	else 
	{
		dx = vx;
		dy = vy;
	}
}

void CouzinAgent::move(double dt)
{
	if (type == DEAD) return;
	act();
	if (type == DEAD) return;
	double sigma = rng->normal(0, 0.25);
		
	double desiredAng = _angle(dx, dy, vx, vy);
	if (desiredAng > 180.0) desiredAng -= 360.0;
	if (turnRate * dt < fabs(desiredAng))
		desiredAng = copysign(turnRate*dt, desiredAng);
	
	desiredAng += sigma;
	//if (desiredAng < 0.0) desiredAng += 360.0;
		
	_rotate(desiredAng);
	
	x += vx*dt;
	y += vy*dt;
	
	if (x < settings.centerX - domainSize/2) x += domainSize;
	if (x > settings.centerX + domainSize/2) x -= domainSize;
	if (y < settings.centerY - domainSize/2) y += domainSize;
	if (y > settings.centerY + domainSize/2) y -= domainSize;
}

#ifdef _RL_VIZ
void CouzinAgent::paint()
{
	if (type != DEAD) _drawArrow (d,   x, y, vx, vy, IvI, 0, 1, 0);
	else			  _drawSphere(d/3, x, y, 1, 0, 0);

}
#endif



