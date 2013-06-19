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

Agent::Agent(double newLearningInterval, Types newType, string newName):
learningInterval(newLearningInterval), lastLearned(0), type(newType), name(newName), id(0)
{
	id = idCount.fetch_add(1);
};





