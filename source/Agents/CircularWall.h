/*
 *  CircularWall.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 06.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Agent.h"
#include "../Misc.h"

class CircularWall : public Agent
{
public:
	double x, y, d;
	
	CircularWall(double newX, double newY, double newD) : Agent(0, IDLER, "CircularWall"), x(newX), y(newY), d(newD)
	{
		sInfo.dim = 0;
		actInfo.dim = 0;
	}
		
	void   move(double dt)     {};
	
#ifdef _RL_VIZ
	void   paint()
	{
		_drawFullCircle(d/2.0, x, y, 0.7, 0.7, 0.7);
	}
	
#endif	

};