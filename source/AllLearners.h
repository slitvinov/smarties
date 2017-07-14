/*
 *  main.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "Learners/NFQ.h"
#include "Learners/NAF.h"
#include "Learners/DPG.h"
#include "Learners/RACER.h"
#include "Learners/DACER.h"
#include "Learners/GAE.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

inline Learner* createLearner(MPI_Comm mastersComm, Environment*const env, Settings&settings)
{
	if(settings.learner=="DQ" || settings.learner=="DQN" || settings.learner=="NFQ") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		settings.nnOutputs = env->aI.maxLabel;
		return new NFQ(mastersComm, env, settings);
	}
	else if (settings.learner == "RACER") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		settings.nnOutputs = RACER::getnOutputs(env->aI.dim);
		#ifdef FEAT_CONTROL
		settings.nnOutputs +=  ContinuousSignControl::addRequestedOutputs(env->aI.dim,env->sI.dimUsed);
		#endif
		return new RACER(mastersComm, env, settings);
	}
	else if (settings.learner == "DACER") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		settings.nnOutputs = DACER::getnOutputs(env->aI.maxLabel);
		return new DACER(mastersComm, env, settings);
	}
	else if (settings.learner == "NA" || settings.learner == "NAF") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		settings.nnOutputs = 1 + NAF::compute_nL(env->aI.dim) + env->aI.dim;
		#ifdef FEAT_CONTROL
		settings.nnOutputs +=  ContinuousSignControl::addRequestedOutputs(env->aI.dim,env->sI.dimUsed);
		#endif
		return new NAF(mastersComm, env, settings);
	}
	else if (settings.learner == "DP" || settings.learner == "DPG") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		settings.nnOutputs = env->aI.dim;
		return new DPG(mastersComm, env, settings);
	}
	else if (settings.learner == "GAE") {
		settings.nnInputs = env->sI.dimUsed*(1+settings.appendedObs);
		settings.nnOutputs = env->aI.dim;
		settings.bSampleSequences = true;
		return new GAE(mastersComm, env, settings);
	} else die("Learning algorithm not recognized\n");
	assert(false);
	return new NFQ(mastersComm, env, settings); //fake, to silence warnings
}
