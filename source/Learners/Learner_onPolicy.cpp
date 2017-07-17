/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner_onPolicy.h"

void Learner_onPolicy::spawnTrainTasks() {} //do nothing

void Learner_onPolicy::prepareData() //cannot call from omp parallel region
{
	assert(work_actions.size() == batchSize);
	assert(work_rewards.size() == batchSize);
	assert(work_assign.size() == batchSize);
	assert(work_done.size() == batchSize);
	assert(work.size() == batchSize);
	for (Uint i = 0; i < batchSize; i++) {
		net->deallocateUnrolledActivations(work[i]);
		assert(work[i]->size() == 0);
		work_actions[i]->clear();
		work_rewards[i]->clear();
		work_assign[i] = -1;
		work_done[i] = 0;
	}
}

bool Learner_onPolicy::batchGradientReady() //are all workspaces filled?
{
	bool done = true;
	for (Uint i = 0; i < batchSize && done; i++)
		done = done && work_done[i];
	return done;
}

int Learner_onPolicy::readyForAgent(const int slave, const int agent)
{
	int ret   = retrieveAssignment(agent);
	int avail = checkFirstAvailable();

	if(ret<0 && avail>=0) //then i can assign an other workspace
	{
		assert(avail%nAgentsPerSlave == 0); //assignment is done slave-wise
		const int islave = agent/nAgentsPerSlave; //actually slave-1

		for(Uint i=islave*nAgentsPerSlave; i<(islave+1)*nAgentsPerSlave; i++)
		{
			if(retrieveAssignment(i) >= 0)
				die("FATAL Starting a new agent before terminating all others on a given slave with an on-policy algo is not supported.\n");
			if(static_cast<int>(i)==agent) ret = avail;
			assert(avail < static_cast<int>(work_assign.size()));
			assert(work[avail]->size() == 0);
			assert(work_assign[avail]==-1 && work_done[avail]==0);
			work_assign[avail++] = i; //assign
			assert(retrieveAssignment(i) == avail-1);
		};
		assert(ret>=0);
		assert(retrieveAssignment(agent) == ret);
	}
	//if here avail<0: nothing is left, wait for ones in progress and apply grad
	return ret;
}

int Learner_onPolicy::slaveHasUnfinishedSeqs(const int slave) const
{
	for(Uint i=slave*nAgentsPerSlave; i<(slave+1)*nAgentsPerSlave; i++)
		if(retrieveAssignment(i)>=0)
			return 1;
	return 0;
}
