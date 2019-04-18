//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Environment.h"
#include "../Network/Builder.h"

Environment::Environment()
{
  descriptors.emplace_back( std::make_unique<MDPdescriptor>() );
}

void Environment::synchronizeEnvironments(
  const std::function<void(void*, size_t)>& sendRecvFunc,
  const Uint nCallingEnvironments)
{
  if(bFinalized) die("Cannot synchronize env description multiple times");
  bFinalized = true;

  sendRecvFunc(&nAgentsPerEnvironment, 1 * sizeof(Uint) );
  sendRecvFunc(&bAgentsHaveSeparateMDPdescriptors, 1 * sizeof(bool) );

  //sendRecvFunc(&nMPIranksPerEnvironment, 1 * sizeof(Uint) );
  //if(nMPIranksPerEnvironment <= 0) {
  //  warn("Overriding nMPIranksPerEnvironment -> 1");
  //  nMPIranksPerEnvironment = 1;
  //}

  bTrainFromAgentData.resize(nAgentsPerEnvironment, true);
  sendRecvFunc(&bTrainFromAgentData.data(), nAgentsPerEnvironment*sizeof(bool));

  nAgents = nAgentsPerEnvironment * nCallingEnvironments;
  assert(nCallingEnvironments>0);

  initDescriptors(bAgentsHaveSeparateMDPdescriptors);

  assert(agents.size() == 0);
  agents.clear();
  agents.reserve(nAgents);
  for(Uint i=0; i<nAgents; i++) {
    // contiguous agents belong to same environment
    const Uint workerID = i / nAgentsPerEnvironment;
    const Uint localID  = i % nAgentsPerEnvironment;
    // agent with the same ID on different environment have the same MDP
    const Uint descriptorID = i % nDescriptors;
    MDPdescriptor& D = descriptors[descriptorID].get();
    agents.emplace_back( std::make_unique<Agent>(i, D, workerID, localID) );
    agents[i]->trackSequence = bTrainFromAgentData[localID];
  }
}

void Environment::initDescriptors(const bool areDifferent)
{
  bAgentsHaveSeparateMDPdescriptors = areDifferent;
  Uint nDescriptors = areDifferent? nAgentsPerEnvironment : 1;

  if(descriptors.size() > nDescriptors) die("conflicts in problem definition");

  descriptors.reserve(nDescriptors);
  for(Uint i=descriptors.size(); i<nDescriptors; ++i)
    descriptors.emplace_back( std::make_unique<MDPdescriptor>() );
}

Uint Environment::getNumberRewardParameters() {
  return 0;
}

// compute the reward given a certain state and param vector
Real Environment::getReward(const std::vector<memReal> s, const Rvec params)
{
  return 0;
}

// compute the gradient of the reward
Rvec Environment::getRewardGrad(const std::vector<memReal> s, const Rvec params)
{
  return Rvec();
}
