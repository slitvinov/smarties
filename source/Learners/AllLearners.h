//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_AllLearners_h
#define smarties_AllLearners_h

//#include "DPG.h"
#include "RACER.h"
#include "Learner_pytorch.h"
#include <fstream>
#include <sstream>

namespace smarties
{

/*
#include "DQN.h"
#include "ACER.h"
#include "NAF.h"
#include "CMALearner.h"
#include "PPO.h"
*/

inline static void printLogfile(std::ostringstream& o, std::string fname, int rank)
{
  if(rank != 0) return;
  std::ofstream fout(fname.c_str(), std::ios::app);
  fout << o.str() << std::endl;
  fout.flush();
  fout.close();
}

inline std::unique_ptr<Learner> createLearner(
  MDPdescriptor& MDP, Settings& settings, DistributionInfo& distrib
)
{
  const ActionInfo aInfo = ActionInfo(MDP);
  std::unique_ptr<Learner> ret;
  std::ostringstream o;
  o << MDP.dimState << " ";

  if (settings.learner == "PYTORCH")
  {
    ret = std::make_unique<Learner_pytorch>(MDP, settings, distrib);
  }
  else
  if (settings.learner == "RACER")
  {
    if(MDP.bDiscreteActions) {
      using RACER_discrete = RACER<Discrete_advantage, Discrete_policy, Uint>;
      MDP.policyVecDim = RACER_discrete::getnDimPolicy(&aInfo);
      o << MDP.maxActionLabel << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<RACER_discrete>(MDP, settings, distrib);
    } else {
      //using RACER_continuous = RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;
      using RACER_continuous = RACER<Param_advantage,Gaussian_policy,Rvec>;
      MDP.policyVecDim = RACER_continuous::getnDimPolicy(&aInfo);
      o << MDP.dimAction << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<RACER_continuous>(MDP, settings, distrib);
    }
  }
  else
  if (settings.learner == "VRACER")
  {
    if(MDP.bDiscreteActions) {
      // V-RACER with discrete actions makes little sense: we revert to RACER
      warn("V-RACER makes little sense with discrete action-spaces. Code will "
        "override user and add parameterization for action advantage.");
      using RACER_discrete = RACER<Discrete_advantage, Discrete_policy, Uint>;
      MDP.policyVecDim = RACER_discrete::getnDimPolicy(&aInfo);
      o << MDP.maxActionLabel << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<RACER_discrete>(MDP, settings, distrib);
    } else {
      //using RACER_continuous = VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;
      using RACER_continuous = RACER<Zero_advantage, Gaussian_policy, Rvec>;
      MDP.policyVecDim = RACER_continuous::getnDimPolicy(&aInfo);
      o << MDP.dimAction << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = std::make_unique<RACER_continuous>(MDP, settings, distrib);
    }
  }
  /*
  else
  if(settings.learner=="NFQ" || settings.learner=="DQN") {
    assert(aInfo.discrete);
    o << MDP.maxActionLabel << " " << MDP.maxActionLabel;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    MDP.policyVecDim = MDP.maxActionLabel;
    ret = new DQN(env, settings);
  }
  else if (settings.learner == "ACER")
  {
    settings.bSampleSequences = true;
    assert(aInfo.discrete == false);
    MDP.policyVecDim = ACER::getnDimPolicy(&aInfo);
    o << MDP.dimAction << " " << MDP.policyVecDim;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    ret = new ACER(env, settings);
  }
  else
  if (settings.learner == "NA" || settings.learner == "NAF")
  {
    settings.bSampleSequences = false;
    MDP.policyVecDim = 2*MDP.dimAction;
    assert(not aInfo.discrete);
    o << MDP.dimAction << " " << MDP.policyVecDim;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    ret = new NAF(env, settings);
  }
  else
  if (settings.learner == "DP" || settings.learner == "DPG")
  {
    settings.bSampleSequences = false;
    MDP.policyVecDim = 2*MDP.dimAction;
    // non-NPER DPG is unstable with annealed network learn rate
    // because critic network must adapt quickly
    if(settings.clipImpWeight<=0) settings.epsAnneal = 0;
    o << MDP.dimAction << " " << MDP.policyVecDim;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    ret = new DPG(env, settings);
  }
  else
  if (settings.learner == "RETPG")
  {
    settings.bSampleSequences = false;
    MDP.policyVecDim = 2*MDP.dimAction;
    // non-NPER DPG is unstable with annealed network learn rate
    // because critic network must adapt quickly
    if(settings.clipImpWeight<=0) settings.epsAnneal = 0;
    o << MDP.dimAction << " " << MDP.policyVecDim;
    printLogfile(o, "problem_size.log", distrib.world_rank);
    ret = new RETPG(env, settings);
  }
  else
  if (settings.learner == "CMA")
  {
    settings.batchSize_loc = settings.batchSize;
    if(settings.ESpopSize<2)
      die("Must be coupled with CMA. Set ESpopSize>1");
    if( settings.nWorkers % settings.learner_size )
      die("nWorkers must be multiple of learner ranks");
    if( settings.ESpopSize % settings.learner_size )
      die("CMA pop size must be multiple of learners");
    if( settings.ESpopSize % settings.nWorkers )
      die("CMA pop size must be multiple of nWorkers");

    if(aInfo.discrete) {
      using CMA_discrete = CMALearner<Uint>;
      MDP.policyVecDim = CMA_discrete::getnDimPolicy(&aInfo);
      o << MDP.maxActionLabel << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = new CMA_discrete(env, settings);
    } else {
      using CMA_continuous = CMALearner<Rvec>;
      MDP.policyVecDim = CMA_continuous::getnDimPolicy(&aInfo);
      if(settings.explNoise > 0) MDP.policyVecDim += MDP.dimAction;
      o << MDP.dimAction << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = new CMA_continuous(env, settings);
    }
  }
  else
  if (settings.learner == "GAE" || settings.learner == "PPO")
  {
    settings.bSampleSequences = false;
    if(aInfo.discrete) {
      using PPO_discrete = PPO<Discrete_policy, Uint>;
      MDP.policyVecDim = PPO_discrete::getnDimPolicy(&aInfo);
      o << MDP.maxActionLabel << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = new PPO_discrete(env, settings);
    } else {
      using PPO_continuous = PPO<Gaussian_policy, Rvec>;
      MDP.policyVecDim = PPO_continuous::getnDimPolicy(&aInfo);
      o << MDP.dimAction << " " << MDP.policyVecDim;
      printLogfile(o, "problem_size.log", distrib.world_rank);
      ret = new PPO_continuous(env, settings);
    }
  }
  */
  else die("Learning algorithm not recognized");

  return ret;
}

}
#endif
