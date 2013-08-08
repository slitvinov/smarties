/*
 *  CouzinDipoleEnvironment.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 18.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "CouzinDipoleEnvironment.h"

void CouzinDipoleEnvironment::evolve(double t)
{
	CouzinEnvironment::evolve(t);
	PotentialFluidEnvironment::getVelocities();
	tm = t;
}