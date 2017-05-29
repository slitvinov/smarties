/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learners/Learner.h"
#include "Learners/NFQ.h"
#include "Learners/NAF.h"
#include "Learners/DPG.h"
#include "Learners/RACER.h"
#include "Learners/DACER.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
Learner* createLearner(MPI_Comm mastersComm, Environment*const env, Settings&settings);

Learner* createLearner(MPI_Comm mastersComm, Environment*const env, Settings&settings)
{
	if(settings.learner=="DQ" || settings.learner=="DQN" || settings.learner=="NFQ") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		settings.nnOutputs = env->aI.maxLabel;
		return new NFQ(mastersComm, env, settings);
	}
	else if (settings.learner == "RACER") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		settings.nnOutputs = RACER::getnOutputs(env->aI.dim);
		settings.separateOutputs = true; //else it does not really work
		return new RACER(mastersComm, env, settings);
	}
	else if (settings.learner == "DACER") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		const int nA = env->aI.maxLabel;
		printf("Read %d outputs\n",nA);
		settings.nnOutputs = DACER::getnOutputs(nA);
		settings.separateOutputs = true; //else it does not really work
		return new DACER(mastersComm, env, settings);
	}
	else if (settings.learner == "NA" || settings.learner == "NAF") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		const int nA = env->aI.dim;
		const int nL = (nA*nA+nA)/2;
		settings.nnOutputs = 1+nL+nA;
		settings.separateOutputs = true; //else it does not really work
		return new NAF(mastersComm, env, settings);
	}
	else if (settings.learner == "DP" || settings.learner == "DPG") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs) + env->aI.dim;
		settings.nnOutputs = 1;
		return new DPG(mastersComm, env, settings);
	} else die("Learning algorithm not recognized\n");
	assert(false);
	return new NFQ(mastersComm, env, settings); //fake, to silence warnings
}
