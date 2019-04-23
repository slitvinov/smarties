//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Communicator.h"
#include "../Utils/SocketsLib.h"

#ifndef SMARTIES_LIB
#include "../Core/Worker.h"
#endif

namespace smarties
{

Communicator::Communicator(int number_of_agents)
{
  set_num_agents(number_of_agents);
  //std::random_device RD;
  //gen = std::mt19937(RD);
  SOCK.server = SOCKET_clientConnect();
}

Communicator::Communicator(int stateDim, int actionDim, int number_of_agents)
{
  set_num_agents(number_of_agents);
  set_state_action_dims(stateDim, actionDim);
  //std::random_device RD;
  //gen = std::mt19937(RD);
  SOCK.server = SOCKET_clientConnect();
}

void Communicator::set_state_action_dims(const int dimState,
                                         const int dimAct,
                                         const int agentID)
{
  if(ENV.bFinalized)
    die("Cannot edit env description after having sent first state.");
  if( (size_t) agentID >= ENV.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");
  ENV.descriptors[agentID]->dimState = dimState;
  ENV.descriptors[agentID]->dimAction = dimAct;
}

void Communicator::set_action_scales(const std::vector<double> uppr,
                                     const std::vector<double> lowr,
                                     const bool bound,
                                     const int agentID)
{
  set_action_scales(uppr,lowr, std::vector<bool>(uppr.size(),bound), agentID);
}
void Communicator::set_action_scales(const std::vector<double> upper,
                                     const std::vector<double> lower,
                                     const std::vector<bool>   bound,
                                     const int agentID)
{
  if(ENV.bFinalized)
    die("Cannot edit env description after having sent first state.");
  if(agentID >= ENV.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");
  if(upper.size() not_eq ENV.descriptors[agentID]->dimAction or
     lower.size() not_eq ENV.descriptors[agentID]->dimAction or
     bound.size() not_eq ENV.descriptors[agentID]->dimAction )
    die("size mismatch");

  ENV.descriptors[agentID]->bDiscreteActions = false;
  ENV.descriptors[agentID]->upperActionValue =
                Rvec(upper.begin(), upper.end());
  ENV.descriptors[agentID]->lowerActionValue =
                Rvec(lower.begin(), lower.end());
  ENV.descriptors[agentID]->bActionSpaceBounded =
    std::vector<int>(bound.begin(), bound.end());
}

void Communicator::set_action_options(const int options,
                                      const int agentID)
{
  set_action_options(std::vector<int>(1, options), agentID);
}

void Communicator::set_action_options(const std::vector<int> options,
                                      const int agentID)
{
  if(ENV.bFinalized)
    die("Cannot edit env description after having sent first state.");
  if(agentID >= ENV.descriptors.size())
    die("Attempted to write to uninitialized MDPdescriptor");
  if(options.size() not_eq ENV.descriptors[agentID]->dimAction)
    die("size mismatch");

  ENV.descriptors[agentID]->bDiscreteActions = true;
  ENV.descriptors[agentID]->discreteActionValues =
    std::vector<Uint>(options.begin(), options.end());
}

void Communicator::set_state_observable(const std::vector<bool> observable,
                                        const int agentID)
{
  if(ENV.bFinalized) {
    printf("ABORTING: cannot edit env description after having sent first state."); fflush(0); abort();
  }
  if(agentID >= ENV.descriptors.size()) {
    printf("ABORTING: Attempted to write to uninitialized MDPdescriptor."); fflush(0); abort();
  }
  if(observable.size() not_eq ENV.descriptors[agentID]->dimState) {
    printf("ABORTING: size mismatch when defining observed/hidden state variables."); fflush(0); abort();
  }

  ENV.descriptors[agentID]->bStateVarObserved =
    std::vector<int>(observable.begin(), observable.end());
}

void Communicator::set_state_scales(const std::vector<double> upper,
                                    const std::vector<double> lower,
                                    const int agentID)
{
  if(ENV.bFinalized) {
    printf("ABORTING: cannot edit env description after having sent first state."); fflush(0); abort();
  }
  if(agentID >= ENV.descriptors.size()) {
    printf("ABORTING: Attempted to write to uninitialized MDPdescriptor."); fflush(0); abort();
  }
  if(upper.size() not_eq ENV.descriptors[agentID]->dimState or
     lower.size() not_eq ENV.descriptors[agentID]->dimState ) {
    printf("ABORTING: upper/lower size mismatch."); fflush(0); abort();
  }
  // For consistency with action space we ask user for a rough box of state vars
  // but in reality we scale with mean and stdev computed during training.
  // This function serves only as an optional initialization for statistiscs.
  Rvec meanState(upper.size()), diffState(upper.size());
  for (Uint i=0; i<upper.size(); ++i) {
    meanState[i] = (upper[i]+lower[i])/2;
    diffState[i] = std::fabs(upper[i]-lower[i]);
  }
  ENV.descriptors[agentID]->stateMean   = meanState;
  ENV.descriptors[agentID]->stateStdDev = diffState;
}

void Communicator::set_num_agents(int _nAgents)
{
  ENV.nAgentsPerEnvironment = _nAgents;
}

void Communicator::env_has_distributed_agents()
{
  /*
  if(comm_inside_app == MPI_COMM_NULL) {
    printf("ABORTING: Distributed agents has no effect on single-process "
    " applications. It means that each simulation rank holds different agents.");
    fflush(0); abort();
    bEnvDistributedAgents = false;
    return;
  }
  */
  if(ENV.bAgentsHaveSeparateMDPdescriptors) {
    printf("ABORTING: Smarties supports either distributed agents (ie each "
    "worker holds some of the agents) or each agent defining a different MDP "
    "(state/act spaces)."); fflush(0); abort();
  }
  bEnvDistributedAgents =  true;
}

void Communicator::agents_define_different_MDP()
{
  if(bEnvDistributedAgents) {
    printf("ABORTING: Smarties supports either distributed agents (ie each "
    "worker holds some of the agents) or each agent defining a different MDP "
    "(state/act spaces)."); fflush(0); abort();
  }
  ENV.initDescriptors(true);
}

void Communicator::disableDataTrackingForAgents(int agentStart, int agentEnd)
{
  ENV.bTrainFromAgentData.resize(ENV.nAgentsPerEnvironment, 1);
  for(int i=agentStart; i<agentEnd; i++)
    ENV.bTrainFromAgentData[i] = 0;
}

void Communicator::sendState(const int agentID, const episodeStatus status,
    const std::vector<double>& state, const double reward)
{
  if ( not ENV.bFinalized ) synchronizeEnvironments();
  const auto& MDP = ENV.getDescriptor(agentID);
  assert(agentID>=0 && agentID<agents.size());
  agents[agentID]->update(status, state, reward);
  agents[agentID]->packStateMsg(BUFF[0]->dataStateBuf);

#ifdef MPI_VERSION
  if(worker not_eq nullptr)
  {
    worker->stepSocketToMaster();
  }
  else
#endif
  {
    SOCKET_Bsend(BUFF[0]->dataStateBuf,  BUFF[0]->sizeStateMsg,  SOCK.server);
    SOCKET_Brecv(BUFF[0]->dataActionBuf, BUFF[0]->sizeActionMsg, SOCK.server);
  }

  agents[agentID]->unpackActionMsg(BUFF[0]->dataActionBuf);

  // we cannot control application. if we received a termination signal we abort
  if(agents[agentID]->learnStatus == KILL) {
    printf("ABORTING: App recvd end-of-training signal.\n"); fflush(0); abort();
  }
}

const std::vector<double>& Communicator::recvAction(const int agentID) const
{
  assert( agents[agentID]->agentStatus < TERM && "Application read action for "
    "a terminal state or truncated episode. Undefined behavior.");
  return agents[agentID]->action;
}

void Communicator::synchronizeEnvironments()
{
  if ( ENV.bFinalized ) return;

#ifdef SMARTIES_INTERNAL
  if(worker not_eq nullptr)
  {
    worker->synchronizeEnvironments();
  }
  else
#endif
  {
    const auto sendBufferFunc = [&](void* buffer, size_t size) {
      SOCKET_Bsend(buffer, size, SOCK.server);
    };
    ENV.synchronizeEnvironments(sendBufferFunc);
    // application process only needs one communication buffer:
    initOneCommunicationBuffer();
  }
  assert(BUFF.size() > 0);
}

void Communicator::initOneCommunicationBuffer()
{
  const Uint nAgents = ENV.nAgentsPerEnvironment;
  Uint maxDimState  = 0, maxDimAction = 0;
  for(size_t i=0; i<ENV.descriptors.size(); ++i)
  {
    maxDimState  = std::max(maxDimState,  ENV.descriptors[i]->dimState );
    maxDimAction = std::max(maxDimAction, ENV.descriptors[i]->dimAction);
  }
  assert(nAgents > 0 && maxDimAction > 0); // state can be 0-D
  BUFF.emplace_back(std::make_unique<COMM_buffer>(maxDimState, maxDimAction) );
}

std::mt19937& Communicator::getPRNG() {
  return gen;
}
bool Communicator::isTraining() {
  return bTrain;
}
int Communicator::desiredNepisodes() {
  return nEpisodes;
}

#ifndef SMARTIES_LIB
#ifndef MPI_VERSION
  #error "Defined SMARTIES_INTERNAL and not MPI_VERSION"
#endif


Communicator::Communicator(Worker*const W, std::mt19937&G, bool isTraining) :
worker(*W), gen(G()), bTrain(isTraining) {}

}
