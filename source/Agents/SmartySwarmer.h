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
#include "FluidAgent.h"
#include "../Misc.h"

class SwarmEnvironment;
#include "../Environments/Environment.h"
#include "../Environments/SwarmEnvironment.h"

enum MovingStates {NORMAL, FAST, SLOW, TURN};

class SmartySwarmer : public FluidAgent
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
	SwarmEnvironment* environment;
	
public:
	double x, y, vx, vy, IvI;
	double physX, physY;
	double domainSize;
	
	double alpha, l, d;
	double gamma, gammaA, gammaT;
	double zoo, zoa;
	double dx, dy;
	MovingStates movState;
	
	SmartySwarmer* closestNeighbour;
	
	RNG* rng;
	
	SmartySwarmer(double newX, double newY, double newD,  double newT, double newDomainSize,
				  double newZoo, double newZoa, double newIvI,  double newAlpha, RNG* newRng);
	
	void computeVecs();
	void move(double dt);
	void setEnvironment(Environment* env);
	
	void   getState(State& s);
	double getReward();
	void   act(Action a);
	
#ifdef _RL_VIZ
	void   paint();
#endif
};