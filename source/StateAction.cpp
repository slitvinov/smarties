/*
 *  StateAction.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 03.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <cassert>

#include "StateAction.h"
#include "rng.h"
#include "rng.h"

ActionIterator::ActionIterator(const ActionInfo& newActInfo) : actInfo(newActInfo), currAction(newActInfo), storedAction(newActInfo), rAction(newActInfo)
{
	reset();	
};

Action& ActionIterator::getRand(RNG* rng)
{	
	for (int i=0; i<actInfo.dim; i++)
		rAction.vals[i] = rng->rand_int32() % actInfo.bounds[i];
	
	return rAction;
}
	
	
Action& ActionIterator::next()
{
	int i = 0;
	
	currAction.vals[0]++;
	
	while (currAction.vals[i] >= actInfo.bounds[i])
	{
		assert(i < actInfo.dim - 1);
		currAction.vals[i] = 0;
		currAction.vals[i+1]++;
	}
	
	return currAction;
}
	
bool ActionIterator::done()
{
	for (int i=0; i<actInfo.dim; i++)
		if (currAction.vals[i] != actInfo.bounds[i] - 1) return false;
	
	return true;
}

void ActionIterator::reset()
{
	for (int i=0; i<actInfo.dim; i++)
		currAction.vals[i] = 0;
	if (actInfo.dim > 0) currAction.vals[0] = -1;
}

void ActionIterator::initAct()
{
	for (int i=0; i<actInfo.dim; i++)
        currAction.vals[i] = rand() % actInfo.bounds[i];
	//if (actInfo.dim > 0) currAction.vals[0] = -1;
}

void ActionIterator::memorize()
{
	storedAction = currAction;
}

Action& ActionIterator::recall()
{
	return storedAction;
}

Action& ActionIterator::show()
{
    return currAction;
}



