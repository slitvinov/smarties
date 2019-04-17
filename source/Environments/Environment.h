//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "../Agent.h"
#include "../Communicators/Communicator_internal.h"

class Builder;

struct Environment
{
  Uint nAgents, nAgentsPerEnvironment;
  Uint nDescriptors = 1;
  bool bAgentsHaveSeparateMDPdescriptors = false;
  Uint nMPIranksPerEnvironment = 1;
  bool bFinalized = false;

  std::vector<std::unique_ptr<MDPdescriptor>> descriptors;
  std::vector<std::unique_ptr<Agent>> agents;
  std::vector<bool> bTrainFromAgentData;

  const MDPdescriptor& getDescriptor(int agentID = 0) const {
    if(not bAgentsHaveSeparateMDPdescriptors) agentID = 0;
    return * descriptors[agentID].get();
  }

  Settings * settings = nullptr;

  Environment();

  virtual bool pickReward(const Agent& agent) const;
  virtual bool predefinedNetwork(Builder & input_net) const;

  // for a given environment, size of the IRL reward dictionary
  virtual Uint getNumberRewardParameters();

  // compute the reward given a certain state and param vector
  virtual Real getReward(const std::vector<memReal> s, const Rvec params);

  // compute the gradient of the reward
  virtual Rvec getRewardGrad(const std::vector<memReal> s, const Rvec params);
};
