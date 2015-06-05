/*
 *  Agent.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Agent.h"

atomic_int Agent::idCount(0);

Agent::Agent(double learningInterval, Types type, string name):
learningInterval(learningInterval), lastLearned(0), type(type), name(name), id(0)
{
	id = idCount.fetch_add(1);
};





