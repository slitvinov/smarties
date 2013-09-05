/*
 *  DynamicColumn.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 06.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Agent.h"
#include "../Misc.h"

class DynamicColumn : public Agent
{		
public:
	double x, y, d, time, x0;

	DynamicColumn(double newX, double newY, double newD) : Agent(0, IDLER, "DynamicColumn"), x(newX), y(newY), d(newD)
	{
		sInfo.dim = 0;
		actInfo.dim = 0;
		time = 0;
		x0 = x;
	}
	
	void   move(double dt)     { time += dt; x = x0 + 0 * d/3 * sin(time/200); };
	
#ifdef _RL_VIZ
	void   paint()
	{
		_drawFullCircle(d/2.0, x, y, 0.5, 0.5, 0.5);
	}
	
#endif	
	
};