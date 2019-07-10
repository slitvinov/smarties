//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_CMALearner_h
#define smarties_CMALearner_h

#include "Learner_approximator.h"

namespace smarties
{

template<typename Action_t>
class CMALearner: public Learner_approximator
{

protected:
  const Uint ESpopSize = settings.ESpopSize;
  const Uint nOwnEnvs = distrib.nOwnedEnvironments;
  const Uint nOwnAgents = distrib.nOwnedAgentsPerAlgo;
  const Uint nOwnAgentsPerEnv = nOwnAgents / nOwnEnvs;
  const Uint batchSize_local = settings.batchSize_local;
  const long nSeqPerStep = nOwnAgents * batchSize_local * ESpopSize;

  // counter per each env of how many sims have ended on this generation:
  std::vector<Uint>  indexEndedPerEnv = std::vector<Uint>(nOwnEnvs, 0);
  // counter per each env of how many agents have currently terminated on this
  //   simulation. no agent can restart unless they all have terminated a sim
  std::vector<Uint> curNumEndedPerEnv = std::vector<Uint>(nOwnEnvs, 0);

  std::vector<Rvec> R = std::vector<Rvec>(nOwnEnvs, Rvec(ESpopSize, 0) );

  static std::vector<Uint> count_pol_outputs(const ActionInfo*const aI);
  static std::vector<Uint> count_pol_starts(const ActionInfo*const aI);


  void prepareCMALoss();

  void computeAction(Agent& agent, const Rvec netOutput) const;
  void Train(const MiniBatch&MB,const Uint wID,const Uint bID) const override;

public:
  CMALearner(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_);

  //main training functions:
  void select(Agent& agent) override;
  void setupTasks(TaskQueue& tasks) override;

  bool blockGradientUpdates() const override;
  bool blockDataAcquisition() const override;

  static Uint getnDimPolicy(const ActionInfo*const aI);
};

}
#endif
