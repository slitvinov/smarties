/*
 *  SmartyLauFish.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "../StateAction.h"
#include "Agent.h"

class LauEnvironment;
#include "../Environments/Environment.h"
#include "../Environments/LauEnvironment.h"

class SmartyLauFish : public Agent
{
private:
	LauEnvironment* environment;
	
public:
	SmartyLauFish();
			
	void   getState(State& s);
	double getReward();
	void   act(Action& a);
	void   move(double dt);
	
	void   setEnvironment(::Environment* env);
};