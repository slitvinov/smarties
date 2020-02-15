//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_SAC_h
#define smarties_SAC_h

#include "Learner_approximator.h"

namespace smarties
{

class SAC : public Learner_approximator
{
  const Uint nA = aInfo.dim();
  const Real explNoise = settings.explNoise;
  Rvec DPGfactor = Rvec(nA, 0);
  mutable std::vector<Rvec> SPGm1 = std::vector<Rvec>(nThreads, Rvec(nA, 0));
  mutable std::vector<Rvec> SPGm2 = std::vector<Rvec>(nThreads, Rvec(nA, 0));
  mutable std::vector<Rvec> DPGm1 = std::vector<Rvec>(nThreads, Rvec(nA, 0));
  mutable std::vector<Rvec> DPGm2 = std::vector<Rvec>(nThreads, Rvec(nA, 0));

  Approximator* actor;
  Approximator* critc;

  void Train(const MiniBatch& MB, const Uint wID, const Uint bID) const override;

public:
  SAC(MDPdescriptor&, Settings&, DistributionInfo&);

  void select(Agent& agent) override;
  void setupTasks(TaskQueue& tasks) override;
};

}

#endif // smarties_DPG_h
