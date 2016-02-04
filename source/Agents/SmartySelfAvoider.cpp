/*
 *  SmartySelfAvoider.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <cmath>
#include "SmartySelfAvoider.h"
#include "../ErrorHandling.h"
#include "../Misc.h"

using namespace ErrorHandling;

SmartySelfAvoider::SmartySelfAvoider(Real x, Real y, Real d,  Real t, Real vx,  Real vy):
Agent(t, ACTOR, "SmartySelfAvoider"), x(x), y(y), d(d), vx(vx), vy(vy)
{
	IvI = sqrt(vx*vx + vy*vy);
	crashed = false;
}

void SmartySelfAvoider::setEnvironment(Environment* env)
{
	this->env = static_cast<SelfAvoidEnvironment*> (env);
	environment = env;
}

void SmartySelfAvoider::_rotate(Real dAng)
{
	Real ang = _angle(vx, vy, 1, 0) + dAng;
	
	vx = IvI * cos(2*M_PI * ang / 360.0);
	vy = IvI * sin(2*M_PI * ang / 360.0);
}

void SmartySelfAvoider::getState(State& s)
{	
	s.vals.clear();
	
	// Assume origin is in 0, 0
	Real dist2cen = _dist(x,y, 0, 0);
	
	s.vals.push_back( sInfo.top[0] );//environment->circWall->d/2 - dist2cen - d/2 ); 
	s.vals.push_back( sInfo.top[1] );//_angle(vx, vy, x-x0, y-y0) );
	
	// Dynamic columns
	
	Column c = env->findClosestColumn(this);
	
	Real d = _dist(x,y, get<0>(c), get<1>(c)) - d/2 - get<2>(c)/2;
	s.vals.push_back( d );
	s.vals.push_back( _angle(vx, vy, get<0>(c) - x, get<1>(c) - y) );

	// Neighbours
	
	SmartySelfAvoider* closestNeighbour = env->findClosestNeighbour(this);
	
	if (closestNeighbour == NULL)
	{
		s.vals.push_back(sInfo.top[4]);
		s.vals.push_back(sInfo.top[5]);
	}
	else
	{
		s.vals.push_back( _dist(x, y, closestNeighbour->x, closestNeighbour->y) - d);
		s.vals.push_back( _angle(vx - closestNeighbour->vx, vy - closestNeighbour->vy,  -(x  - closestNeighbour->x),  -(y  - closestNeighbour->y)));
	}
}

Real SmartySelfAvoider::getReward()
{
	Real reward = 0;
	//if (crashed) reward -= 1;
	
	Column c = env->findClosestColumn(this);
	if ( _dist(x,y, get<0>(c), get<1>(c)) - d/2 - get<2>(c)/2 < 0 )
		reward -= 1;
	
	SmartySelfAvoider* closestNeighbour = env->findClosestNeighbour(this);
	if ( closestNeighbour != NULL && _dist(x, y, closestNeighbour->x, closestNeighbour->y) - d < 0 )
		reward -= 1;
	
	return reward;
}

void SmartySelfAvoider::act(Action& a)
{
	Real dAng = 5.0;
	
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

void SmartySelfAvoider::move(Real dt)
{
	Real xOld = x;
	Real yOld = y;
	x += vx*dt;
	y += vy*dt;
	
	bool inside = (_dist(x, y, 0, 0) < env->rWall - d/2);
	
	vx = inside ? vx : -vx;
	vy = inside ? vy : -vy;
	x  = inside ? x  : xOld;
	y  = inside ? y  : yOld;

	crashed = !inside;
}


