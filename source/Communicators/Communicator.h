//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include <vector>
#include <cstring>
#include <random>

#include <mpi.h>

#include "Communicator_utils.h"

struct COMM_buffer;
class Worker;

class Communicator
{
public:
  Communicator(int number_of_agents = 1);
  Communicator(int stateDim, int actionDim, int number_of_agents = 1);

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// BEGINNER METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // Send first state of an episode in order to get first action:
  void sendInitState(const std::vector<double>& state,
                     const int agentID=0)
  {
    return sendState(agentID, INIT, state, 0);
  }

  // Send normal state and reward:
  void sendState(const std::vector<double>& state,
                 const double reward,
                 const int agentID = 0)
  {
    return sendState(agentID, CONT, state, reward);
  }

  // Send terminal state/reward: the last step of an episode which ends because
  // of TERMINATION (e.g. agent cannot continue due to failure or success).
  void sendTermState(const std::vector<double>& state,
                    const double reward,
                    const int agentID = 0)
  {
    return sendState(agentID, TERM, state, reward);
  }

  // Send truncated state/reward: the last step of an episode which ends because
  // of TRUNCATION (e.g. agent cannot continue due to time limits). Difference
  // from TERMINATION is that policy was not direct cause of episode's end.
  void truncateSeq(const std::vector<double>& state,
                   const double reward,
                   const int agentID = 0)
  {
    return sendState(agentID, TRNC, state, reward);
  }

  // receive action for the latest given state:
  const std::vector<double>& recvAction(const int agentID = 0);

  void set_state_action_dims(const int dimState, const int dimAct,
                             const int agentID);

  void set_action_scales(const std::vector<double> uppr,
                         const std::vector<double> lowr,
                         const bool bound,
                         const int agentID);

  void set_action_scales(const std::vector<double> upper,
                         const std::vector<double> lower,
                         const std::vector<bool>   bound,
                         const int agentID);

  void set_action_options(const int options,
                          const int agentID);

  void set_action_options(const std::vector<int> options,
                          const int agentID);

  void set_state_observable(const std::vector<bool> observable,
                            const int agentID);

  void set_state_scales(const std::vector<double> upper,
                        const std::vector<double> lower,
                        const int agentID);

  void set_num_agents(int _nAgents);

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// ADVANCED METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  void env_has_distributed_agents();

  void agents_define_different_MDP();

  void disableDataTrackingForAgents(int agentStart, int agentEnd);

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// UTILITY METHODS ///////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  std::mt19937& getPRNG();
  bool isTraining();
  int desiredNepisodes();

  //////////////////////////////////////////////////////////////////////////////
  ///////////////////////////// DEVELOPER METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
protected:
  bool bEnvDistributedAgents = false;

  Environment ENV;
  std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;
  std::vector<std::unique_ptr<COMM_buffer>> BUFF;

  struct {
    int server;
    std::vector<int> clients;
  } SOCK;

  void synchronizeEnvironments();

  //random number generation:
  std::mt19937 gen;
  //internal counters & flags
  bool bTrain = true;
  int nEpisodes -1;

  // To get access to smarties' internals:
  #ifdef MPI_VERSION
  std::shared_pointer<Worker> worker;
  #endif

  //called by app to interact with smarties:
  void sendState(const int agentID, const episodeStatus status,
    const std::vector<double>& state, const double reward);
  //called as constructor only by child class: Communicator_internal
  Communicator(std::mt19937& G, bool _bTrain, int nEps) :
    gen(G()), bTrain(_bTrain), nEpisodes(nEps) { }
};

struct COMM_buffer
{
  COMM_buffer(const size_t maxStateDim, const size_t maxActionDim) :
    maxStateDim(maxStateDim), maxActionDim(maxActionDim),
    sizeStateMsg(Agent::computeStateMsgSize(maxStateDimension)),
    sizeActionMsg(Agent::computeActionMsgSize(maxActionDimension)),
    dataStateBuf(malloc(sizeStateMsg)), dataActionBuf(malloc(sizeActionMsg)) { }

  ~COMM_buffer() {
    assert(dataStateBuf not_eq nullptr && dataActionBuf not_eq nullptr);
    free(dataActionBuf);
    free(dataStateBuf);
  }

  COMM_buffer(const COMM_buffer& c) = delete;
  COMM_buffer& operator= (const COMM_buffer& s) = delete;

  const size_t maxStateDim, maxActionDim, sizeStateMsg, sizeActionMsg;
  void * const dataStateBuf;
  void * const dataActionBuf;
};
