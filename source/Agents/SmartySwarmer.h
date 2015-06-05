/*
 *  SmartySwarmer.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 09.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include <vector>

#include "../StateAction.h"
#include "Agent.h"
#include "../Misc.h"

class SwarmEnvironment;
#include "../Environments/Environment.h"
#include "../Environments/SwarmEnvironment.h"

enum MovingStates {NORMAL, FAST, SLOW, TURN};

class SmartySwarmer : public Agent
{
	struct Comparator
	{
		double x, y;
		Comparator(double x, double y) : x(x), y(y) {};

		inline bool operator() (SmartySwarmer* a, SmartySwarmer* b)
		{
			const double da = _dist(x, y, a->physX, a->physY);
			const double db = _dist(x, y, b->physX, b->physY);
			return (da < db);
		}
	};

public:
	static const int nVort = 2;
    double vortices[nVort];
    pair<double, double> vortCoos[nVort];
    pair<double, double> vortVels[nVort];

	SwarmEnvironment* env;

public:
	double x, y, vx, vy, IvI;
	double physX, physY;
	double domainSize;

	double alpha, l, d;
	double gamma, gammaA, gammaT;

	SmartySwarmer(double x, double y, double d,  double T, double domainSize, double IvI,  double alpha);

	void move(double dt);
	void setEnvironment(Environment* env);

	void   getState(State& s);
	double getReward();
	void   act(Action& a);
};
