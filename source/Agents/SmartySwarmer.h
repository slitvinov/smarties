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
		Real x, y;
		Comparator(Real x, Real y) : x(x), y(y) {};

		inline bool operator() (SmartySwarmer* a, SmartySwarmer* b)
		{
			const Real da = _dist(x, y, a->physX, a->physY);
			const Real db = _dist(x, y, b->physX, b->physY);
			return (da < db);
		}
	};

public:
	static const int nVort = 2;
    Real vortices[nVort];
    pair<Real, Real> vortCoos[nVort];
    pair<Real, Real> vortVels[nVort];

	SwarmEnvironment* env;

public:
	Real x, y, vx, vy, IvI;
	Real physX, physY;
	Real domainSize;

	Real alpha, l, d;
	Real gamma, gammaA, gammaT;

	SmartySwarmer(Real x, Real y, Real d,  Real T, Real domainSize, Real IvI,  Real alpha);

	void move(Real dt);
	void setEnvironment(Environment* env);

	void   getState(State& s);
	Real getReward();
	void   act(Action& a);
};
