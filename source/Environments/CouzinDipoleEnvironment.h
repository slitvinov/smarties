/*
 *  CouzinDipoleEnvironment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 18.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "CouzinEnvironment.h"
#include "PotentialFluidEnvironment.h"

class CouzinDipoleEnvironment: public CouzinEnvironment, public PotentialFluidEnvironment
{
public:
	CouzinDipoleEnvironment(vector<Agent*> newAgents) : Environment(newAgents), CouzinEnvironment(newAgents), PotentialFluidEnvironment(newAgents) {};
	void evolve(double t);
};