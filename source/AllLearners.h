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
//#include "Learners/DACER.h"
//#include "Learners/CACER.h"
#include "Learners/GAE.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

inline Learner* createLearner(Environment*const env, Settings&settings)
{
  if(settings.learner=="DQN" || settings.learner=="NFQ") {
    assert(env->aI.discrete);
    settings.policyVecDim = env->aI.maxLabel;
    return new NFQ(env, settings);
  }
  else if (settings.learner == "RACER" || settings.learner == "POAC") {
    if(env->aI.discrete) {
      settings.policyVecDim = RACER_disc::getnDimPolicy(&env->aI);
      return new RACER_disc(env, settings);
    } else {
      typedef RACER_experts RACER_continuous;
      //typedef RACER_cont RACER_continuous;
      settings.policyVecDim = RACER_continuous::getnDimPolicy(&env->aI);
      return new RACER_continuous(env, settings);
    }
  }
  else if (settings.learner == "NA" || settings.learner == "NAF") {
    settings.policyVecDim = 2*env->aI.dim;
    assert(not env->aI.discrete);
    return new NAF(env, settings);
  }
  else if (settings.learner == "DP" || settings.learner == "DPG") {
    settings.policyVecDim = 2*env->aI.dim;
    return new DPG(env, settings);
  }
  else if (settings.learner == "GAE" || settings.learner == "PPO") {
    settings.batchSize = ceil(settings.batchSize/(Real)env->nAgentsPerRank);
    printf("Batchsize set to %d\n", settings.batchSize);

    if(env->aI.discrete) {
      settings.policyVecDim = GAE_disc::getnDimPolicy(&env->aI);
      return new GAE_disc(env, settings);
    } else {
      settings.policyVecDim = GAE_cont::getnDimPolicy(&env->aI);
      return new GAE_cont(env, settings);
    }
  } else die("Learning algorithm not recognized\n");
  assert(false);
  return new NFQ(env, settings); //fake, to silence warnings
}
