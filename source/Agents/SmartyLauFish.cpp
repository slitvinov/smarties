/*
 *  SmartyLauFish.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "SmartyLauFish.h"
#include "../ErrorHandling.h"
#include "../Misc.h"

using namespace ErrorHandling;

SmartyLauFish::SmartyLauFish():

Agent(0.1, ACTOR, "SmartyLauFish")
{
}

void SmartyLauFish::setEnvironment(::Environment* env)
{
	environment = static_cast<LauEnvironment*> (env);
}

void SmartyLauFish::getState(State& s)
{	
	s.vals.clear();
    s.vals.push_back(environment->getState());
}

double SmartyLauFish::getReward()
{
    return environment->getReward();
}

void SmartyLauFish::act(Action& a)
{
	environment->act(a.vals[0]);
}

void SmartyLauFish::move(double dt)
{
    environment->fluid->run();
}
