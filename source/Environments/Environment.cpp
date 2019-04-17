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
  const std::function<void(void*, size_t)>& sendRecvFunc )
{
  if(bFinalized) die("Cannot synchronize env description multiple times");
  bFinalized = true;

  sendRecvFunc(&nAgentsPerEnvironment, 1 * sizeof(Uint) );
  sendRecvFunc(&bAgentsHaveSeparateMDPdescriptors, 1 * sizeof(bool) );
  sendRecvFunc(&nMPIranksPerEnvironment, 1 * sizeof(Uint) );
  if(nMPIranksPerEnvironment <= 0) {
    warn("Overriding nMPIranksPerEnvironment -> 1");
    nMPIranksPerEnvironment = 1;
  }

  bTrainFromAgentData.resize(nAgentsPerEnvironment, true);
  sendRecvFunc(&bTrainFromAgentData.data(), nAgentsPerEnvironment*sizeof(bool));

  if(settings not_eq nullptr) {
    nAgents = nAgentsPerEnvironment * settings->nWorkers_own;
    settings->nAgents = nAgents;
  } else {
    nAgents = nAgentsPerEnvironment;
  }

  initDescriptors(bAgentsHaveSeparateMDPdescriptors);

  agents.resize(nAgents, nullptr);
  for(Uint i=0; i<nAgents; i++)
  {
    // contiguous agents belong to same environment
    const Uint workerID = i / nAgentsPerEnvironment;
    const Uint localID  = i % nAgentsPerEnvironment;
    // agent with the same ID on different environment have the same MDP
    const Uint descriptorID = i % nDescriptors;
    MDPdescriptor& descriptor = descriptors[descriptorID].get();
    agents[i] = new Agent(i, descriptor, workerID, localID);
    agents[i]->trackSequence = bTrainFromAgentData[localID];
  }
}

void Environment::initDescriptors(const bool areDifferent)
{
  bAgentsHaveSeparateMDPdescriptors = areDifferent;
  if(areDifferent) nDescriptors = nAgentsPerEnvironment;
  else nDescriptors = 1;

  if( descriptors.size() > nDescriptors)
    die("conflicts in problem definition");

  descriptors.reserve(nDescriptors);
  for(Uint i=descriptors.size(); i<nDescriptors; ++i)
    descriptors.emplace_back( std::make_unique<MDPdescriptor>() );
}

Communicator_internal Environment::create_communicator()
{
  Communicator_internal comm(settings);
  comm_ptr = &comm;

  if(settings.runInternalApp)
  {
    if(settings.workers_rank>0) // aka not a master
    {
      if( (settings.workers_size-1) % settings.workersPerEnv != 0)
        die("Number of ranks does not match app\n");

      int workerGroup = (settings.workers_rank-1) / settings.workersPerEnv;

      MPI_Comm app_com;
      MPI_Comm_split(settings.workersComm, workerGroup, settings.workers_rank,
        &app_com);
      comm.set_application_mpicom(app_com, workerGroup);
      comm.ext_app_run(); //worker rank will remain here for ever
      return comm;
    }
    else // master : unblock creation of app comm
    {
      if(settings.bSpawnApp)
       die("smarties is being told to create environment on worker ranks, but no worker ranks were created. Increase the number of mpi processes!");
      MPI_Comm tmp_com;
      MPI_Comm_split(settings.workersComm, MPI_UNDEFINED, 0, &tmp_com);
    }
  }

  if(settings.bSpawnApp) comm_ptr->launch();



  return comm;
}

Environment::~Environment() {
  for (auto & trash : agents) _dispose_object(trash);
}

bool Environment::predefinedNetwork(Builder & input_net) const
{
  // this function is to be filled by child classes
  // to implement convolutional models
  return false;
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
