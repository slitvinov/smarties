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
#include "Learners/ACER.h"
#include "Learners/GAE.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

inline void print(std::ostringstream& o, std::string fname, int rank)
{
  if(rank != 0) return;
  ofstream fout(fname.c_str(), ios::app);
  fout << o.str() << endl;
  fout.flush();
  fout.close();
}

inline Learner* createLearner(Environment*const env, Settings&settings)
{
  std::ostringstream o;
  o << env->sI.dim << " ";
  if(settings.learner=="DQN" || settings.learner=="NFQ") {
    assert(env->aI.discrete);
    o << env->aI.maxLabel << " " << env->aI.maxLabel;
    print(o, "problem_size.log", settings.world_rank);
    settings.policyVecDim = env->aI.maxLabel;
    return new NFQ(env, settings);
  }
  else if (settings.learner == "RACER" || settings.learner == "POAC") {
    if(env->aI.discrete) {
      settings.policyVecDim = RACER_disc::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      return new RACER_disc(env, settings);
    } else {
      typedef RACER_experts RACER_continuous;
      //typedef RACER_cont RACER_continuous;
      settings.policyVecDim = RACER_continuous::getnDimPolicy(&env->aI);
      o << env->aI.dim << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      return new RACER_continuous(env, settings);
    }
  }
  else if (settings.learner == "ACER") {
    assert(env->aI.discrete == false);
    settings.policyVecDim = ACER::getnDimPolicy(&env->aI);
    o << env->aI.dim << " " << settings.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    return new ACER(env, settings);
  }
  else if (settings.learner == "NA" || settings.learner == "NAF") {
    settings.policyVecDim = 2*env->aI.dim;
    assert(not env->aI.discrete);
    o << env->aI.dim << " " << settings.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    return new NAF(env, settings);
  }
  else if (settings.learner == "DP" || settings.learner == "DPG") {
    settings.policyVecDim = 2*env->aI.dim;
    o << env->aI.dim << " " << settings.policyVecDim;
    print(o, "problem_size.log", settings.world_rank);
    return new DPG(env, settings);
  }
  else if (settings.learner == "GAE" || settings.learner == "PPO") {
    settings.batchSize = ceil(settings.batchSize/(Real)env->nAgentsPerRank);
    printf("Batchsize set to %d\n", settings.batchSize);

    if(env->aI.discrete) {
      settings.policyVecDim = GAE_disc::getnDimPolicy(&env->aI);
      o << env->aI.maxLabel << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      return new GAE_disc(env, settings);
    } else {
      settings.policyVecDim = GAE_cont::getnDimPolicy(&env->aI);
      o << env->aI.dim << " " << settings.policyVecDim;
      print(o, "problem_size.log", settings.world_rank);
      return new GAE_cont(env, settings);
    }
  } else die("Learning algorithm not recognized\n");
  assert(false);
  return new NFQ(env, settings); //fake, to silence warnings
}
