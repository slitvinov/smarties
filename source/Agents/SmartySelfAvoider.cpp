/*
 *  SmartySelfAvoider.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "SmartySelfAvoider.h"
#include "../ErrorHandling.h"
#include "../Misc.h"

using namespace ErrorHandling;

SmartySelfAvoider::SmartySelfAvoider(double newX, double newY, double newD,  double newT, double newVx,  double newVy):

Agent(newT, ACTOR, "SmartySelfAvoider"), x(newX), y(newY), d(newD), vx(newVx), vy(newVy)
{
	IvI = sqrt(vx*vx + vy*vy);
	crashed = false;
	closestDynCol = NULL;
	closestNeighbour = NULL;
}

void SmartySelfAvoider::setEnvironment(Environment* env)
{
	environment = static_cast<SelfAvoidEnvironment*> (env);
}

void SmartySelfAvoider::_rotate(double dAng)
{
	double ang = _angle(vx, vy, 1, 0) + dAng;
	//if (ang > 180.0) ang -= 360.0;
	
	vx = IvI * cos(2*M_PI * ang / 360.0);
	vy = IvI * sin(2*M_PI * ang / 360.0);
}

void SmartySelfAvoider::getState(State& s)
{	
	s.vals.clear();
	// Circular wall
	double x0 = environment->circWall->x;
	double y0 = environment->circWall->y;
	
	double dist2cen = _dist(x,y,x0,y0);
	
	s.vals.push_back( 0 );//environment->circWall->d/2 - dist2cen - d/2 ); 
	s.vals.push_back( 0 );//_angle(vx, vy, x-x0, y-y0) );
	
	// Dynamic columns
	
	if (closestDynCol == NULL) closestDynCol = environment->findClosestDynColumn(this);	
	if (closestDynCol == NULL)
	{
		s.vals.push_back(sInfo.top[2]);
		s.vals.push_back(sInfo.top[3]);
	}
	else
	{
		double min = _dist(x,y, closestDynCol->x, closestDynCol->y) - d/2 - closestDynCol->d/2;
		s.vals.push_back( min );
		s.vals.push_back( _angle(vx, vy, x - closestDynCol->x, y - closestDynCol->y) );
	}	
	
	// Neighbours
	
	if (closestNeighbour == NULL) closestNeighbour = environment->findClosestNeighbour(this);
	
	if (closestNeighbour == NULL)
	{
		s.vals.push_back(sInfo.top[4]);
		s.vals.push_back(sInfo.top[5]);
	}
	else
	{
		s.vals.push_back( _dist(x, y, closestNeighbour->x, closestNeighbour->y) - d);
		s.vals.push_back( _angle(vx - closestNeighbour->vx, vy - closestNeighbour->vy,  (x  - closestNeighbour->x),  (y  - closestNeighbour->y)));	
	}
}

double SmartySelfAvoider::getReward()
{
	double reward = 0;
	//if (crashed) reward -= 1;
	
	if (closestDynCol == NULL) closestDynCol = environment->findClosestDynColumn(this);		
	if ( closestDynCol != NULL &&
		_dist(x, y, closestDynCol->x, closestDynCol->y) - d/2 - closestDynCol->d/2 < 0 )
	{
		reward -= 1;
	}
	
	if (closestNeighbour == NULL) closestNeighbour = environment->findClosestNeighbour(this);
	if ( closestNeighbour != NULL && 
		_dist(x, y, closestNeighbour->x, closestNeighbour->y) - d < 0 )
	{
		reward -= 1;
	}
	
	environment->accumulateReward(reward);
	return reward;
}

void SmartySelfAvoider::act(Action a)
{
	double dAng = 5.0;
	
	switch (a.vals[0])
	{
		case 0:
			// Move on
			break;
			
		case 1:
			// Turn left by 5 deg
			_rotate(dAng);
			break;
			
		case 2:
			// Turn right by 5 deg
			_rotate(-dAng);
			break;
			
		default:
			die("aaaaaAAAAAAAAAAAAAAAAAAA!!!!!!!!!!!!");
			break;
	}
}

void SmartySelfAvoider::move(double dt)
{
	double xOld = x;
	double yOld = y;
	x += vx*dt;
	y += vy*dt;
	
	bool inside = (_dist(x, y, environment->circWall->x, environment->circWall->y) < environment->circWall->d/2 - d/2);
	
	vx = inside ? vx : -vx;
	vy = inside ? vy : -vy;
	x  = inside ? x  : xOld;
	y  = inside ? y  : yOld;
	
	closestDynCol = NULL;
	closestNeighbour = NULL;
	crashed = !inside;
}

#ifdef _RL_VIZ
void SmartySelfAvoider::paint()
{
	_drawSphere(d/2.0, x, y, 0, 1, 0);
	//_drawArrow(d, x, y, vx, vy, IvI, 0, 1, 0);
}
#endif



