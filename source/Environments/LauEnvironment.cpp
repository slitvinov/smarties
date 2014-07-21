/*
 *  LauEnvironment.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "LauEnvironment.h"
#include "../Misc.h"
#include "../ErrorHandling.h"
#include "../Settings.h"
#include "../StateAction.h"

using namespace ErrorHandling;

LauEnvironment::LauEnvironment(vector<Agent*> newAgents, const int argc, const char** argv): ::Environment(newAgents)
{
    fluid = new IF2D_FluidMediatedLau(argc, argv);
}

void LauEnvironment::setDims()
{
	sI.dim = 1;
	sI.type = DISCR;
	
	// dist to wall
	sI.bounds.push_back(10);
	sI.top.push_back(1.0);
	sI.bottom.push_back(0.0);
	sI.aboveTop.push_back(true);
	sI.belowBottom.push_back(true);
	
    aI.dim = 1;
	for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
}

double LauEnvironment::getReward()
{
    return fluid->getReward();
}

double LauEnvironment::getState()
{
    return fluid->getState();
}

void LauEnvironment::act(int a)
{
    return fluid->act(a);
}
