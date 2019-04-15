//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Communicator.h"
#include "Communicator_utils.cpp"
#include <iomanip>
#include <fstream>
#include <iostream>
#include <sstream>

#include <sys/un.h>

Communicator::Communicator(int number_of_agents)
{
  set_num_agents(number_of_agents);
  launch();
}

Communicator::Communicator(int stateDim, int actionDim, int number_of_agents)
{
  set_num_agents(number_of_agents);
  set_state_action_dims(stateDim, actionDim);
  launch();
}

Communicator::commonInit()
{
  std::random_device RD;
  gen = std::mt19937(RD);

  if(ENV.nAgentsPerEnvironment <= 0) {
    printf("FATAL: Must have at least one agent.\n");
    fflush(0); abort();
  }

  connect2server();
}

// smarties side
Communicator::Communicator(std::mt19937& G, bool _bTrain, int nEps) :
  gen(G()), bTrain(_bTrain), nEpisodes(nEps)
{
}

void Communicator::sendState(const int agentID, const episodeStatus status,
    const std::vector<double>& state, const double reward)
{
  if ( not ENV.bFinalized ) synchronizeEnvironments();
  const auto& MDP = ENV.getDescriptor(agentID);
  assert(agentID>=0 && agentID<agents.size());
  agents[agentID]->update(status, state, reward);
  agents[agentID]->packStateMsg(BUFF[0]->dataStateBuf);

  if(worker not_eq nullptr)
  {
    worker->stepSocketToMaster();
  }
  else
  {
    SOCKET_Send(BUFF[0]->dataStateMsg,  BUFF[0]->sizeStateMsg,  SOCK.Socket);
    SOCKET_Recv(BUFF[0]->dataActionMsg, BUFF[0]->sizeActionMsg, SOCK.Socket);
  }

  agents[agentID]->unpackActionMsg(BUFF[0]->dataActionMsg);

  // we cannot control application. if we received a termination signal we abort
  if(agents[agentID]->learnStatus == KILL) die("App recvd kill signal...");

  //if (status >= TERM_COMM) stored_actions[agentID][0] = AGENT_TERMSIGNAL;
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

  if(worker not_eq nullptr)
  {
    worker->synchronizeEnvironments();
  }
  else
  {
    const auto sendBufferFunc = [&](void* buffer, size_t size) {
      SOCKET_Bsend(buffer, size, SOCK.Socket);
    };
    ENV.synchronizeEnvironments(sendBufferFunc);
    finalizeCommunicationBuffers();
  }

  print();
}

void Communicator::finalizeCommunicationBuffers()
{
  const Uint nAgents = ENV.nAgentsPerEnvironment;
  Uint maxDimState  = 0, maxDimAction = 0;
  for(size_t i=0; i<ENV.descriptors.size(); ++i)
  {
    maxDimState  = std::max(maxDimState,  ENV.descriptors[i]->dimState );
    maxDimAction = std::max(maxDimAction, ENV.descriptors[i]->dimAction);
  }
  BUFF.emplace_back(maxStateDim, maxActionDim);
}

void Communicator::connect2server()
{
  if( ( SOCK.server = socket(AF_UNIX, SOCK_STREAM, 0) ) == -1 )
  {
    printf("Socket failed"); fflush(0); abort();
  }

  int _TRU = 1;
  if( setsockopt(SOCK.server, SOL_SOCKET, SO_REUSEADDR, &_TRU, sizeof(int))<0 )
  {
    printf("Sockopt failed\n"); fflush(0); abort();
  }

  // Specify the socket
  char SOCK_PATH[] = "../smarties_AFUNIX_socket_FD";

  // Specify the server
  struct sockaddr_un serverAddress;
  bzero((char *)&serverAddress, sizeof(serverAddress));
  serverAddress.sun_family = AF_UNIX;
  strcpy(serverAddress.sun_path, SOCK_PATH);
  const int servlen = sizeof(serverAddress.sun_family)
                    + strlen(serverAddress.sun_path)+1;

  // Connect to the server
  size_t nAttempts = 0;
  while (connect(SOCK.server, (struct sockaddr *)&serverAddress, servlen) < 0)
  {
    if(++nAttempts % 1000 == 0) {
      printf("Application is taking too much time to connect to smarties."
             " If your application needs to change directories (e.g. set up a"
             " dedicated directory for each run) it should do so AFTER"
             " the connection to smarties has been initialzed.\n");
    }
    usleep(1);
  }
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
