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

#define envInfo  int
#define CONT_COMM 0
#define INIT_COMM 1
#define TERM_COMM 2
#define TRNC_COMM 3
#define FAIL_COMM 4
#define AGENT_KILLSIGNAL -256
#define AGENT_TERMSIGNAL  256

#include <mpi.h>

#include "Communicator_utils.h"

/* // TO KILL
struct MPI_info
{
  // comm to talk to master:
  MPI_Comm comm_learn_pool = MPI_COMM_NULL;
  int rank_learn_pool = -1, size_learn_pool = -1;
  // only for MPI-based *applications* eg. flow solvers:
  MPI_Comm comm_inside_app = MPI_COMM_NULL;
  int rank_inside_app = -1, size_inside_app = -1;
  // for MPI-based applications, to split simulations between groups of ranks
  // each learner can have multiple mpi groups of workers
  int workerGroup = -1;

  MPI_Request send_request = MPI_REQUEST_NULL;
  MPI_Request recv_request = MPI_REQUEST_NULL;
};
*/

class Communicator
{
protected:
  bool bEnvDistributedAgents   = false;

  Environment ENV;
  std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;

  struct {
    int server;
    std::vector<int> clients;
  } SOCK;

  std::vector<std::unique_ptr<COMM_buffer>> BUFF;

  //random number generation:
  std::mt19937 gen;

  //internal counters & flags
  unsigned long iter = 0;
  //App output file descriptor:
  int fd; fpos_t pos;

  bool bTrain = true;
  int nEpisodes -1;

 public:
  std::mt19937& getPRNG();
  bool isTraining();
  int desiredNepisodes();

  //called by app to interact with smarties
  void sendState(const int iAgent, const envInfo status,
    const std::vector<double> state, const double reward);

  // specialized functions:
  // initial state: the first of an ep. By definition 0 reward (no act done yet)
  inline void sendInitState(const std::vector<double> state, const int iAgent=0)
  {
    return sendState(iAgent, INIT_COMM, state, 0);
  }
  // terminal state: the last of a sequence which ends because a TERMINAL state
  // has been encountered (ie. agent cannot continue due to failure)
  inline void sendTermState(const std::vector<double> state, const double reward, const int iAgent = 0)
  {
    return sendState(iAgent, TERM_COMM, state, reward);
  }
  // truncation: usually episode can be over due to time constrains
  // it differs because the policy was not at fault for ending the episode
  // meaning that episode could continue following the policy
  // and that the value of this last state should be the expected on-pol returns
  inline void truncateSeq(const std::vector<double> state, const double reward, const int iAgent = 0)
  {
    return sendState(iAgent, TRNC_COMM, state, reward);
  }
  // `normal` state inside the episode
  inline void sendState(const std::vector<double> state, const double reward, const int iAgent = 0)
  {
    return sendState(iAgent, CONT_COMM, state, reward);
  }
  // receive action sent by smarties
  std::vector<double> recvAction(const int iAgent = 0);

  Communicator(const int socket, const int state_components, const int action_components, const int number_of_agents = 1);
  Communicator(int socket, bool spawn, std::mt19937& G, int _bTr, int nEps);
  virtual ~Communicator();

  virtual void connect2server();

 protected:

  //Communication over sockets


  void print();

  void env_has_distributed_agents()
  {
    if(comm_inside_app == MPI_COMM_NULL) {
      die("Distributed agents has no effect on single-process applications. "
           "It means that each simulation rank holds different agents.");
      bEnvDistributedAgents = false;
      return;
    }
    if(ENV.bAgentsHaveSeparateMDPdescriptors) {
      die("support either distributed agents (ie each worker holds some of the "
          "agents) or each agent defining a different MDP (state/act spaces).");
    }
    bEnvDistributedAgents =  true;
  }

  void agents_define_different_MDP()
  {
    if(bEnvDistributedAgents) {
      die("support either distributed agents (ie each worker holds some of the "
          "agents) or each agent defining a different MDP (state/act spaces).");
    }
    ENV.initDescriptors(true);
  }

  void set_state_action_dims(const int dimState, const int dimAct,
                             const int agentID)
  {
    if(ENV.bFinalized)
      die("Cannot edit env description after having sent first state.");
    if( (size_t) agentID >= ENV.descriptors.size())
      die("Attempted to write to uninitialized MDPdescriptor");
    ENV.descriptors[agentID]->dimState = dimState;
    ENV.descriptors[agentID]->dimAct = dimAct;
  }

  void set_state_observable(const std::vector<bool> observable,
                            const int agentID)
  {
    if(ENV.bFinalized)
      die("Cannot edit env description after having sent first state.");
    if(agentID >= ENV.descriptors.size())
      die("Attempted to write to uninitialized MDPdescriptor");
    if(observable.size() not_eq ENV.descriptors[agentID]->dimState)
      die("size mismatch when defining observed/hidden state variables");

    ENV.descriptors[agentID]->bStateVarObserved = observable;
  }

  void set_state_scales(const std::vector<double> upper,
                        const std::vector<double> lower,
                        const int agentID)
  {
    if(ENV.bFinalized)
      die("Cannot edit env description after having sent first state.");
    if(agentID >= ENV.descriptors.size())
      die("Attempted to write to uninitialized MDPdescriptor.");
    if(upper.size() not_eq ENV.descriptors[agentID]->dimState or
       lower.size() not_eq ENV.descriptors[agentID]->dimState )
      die("size mismatch");
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

  void set_action_scales(const std::vector<double> uppr,
                         const std::vector<double> lowr,
                         const bool bound,
                         const int agentID)
  {
    set_action_scales(uppr,lowr, std::vector<bool>(uppr.size(),bound), agentID);
  }
  void set_action_scales(const std::vector<double> upper,
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
    ENV.descriptors[agentID]->bActionSpaceBounded = bound;
  }

  void set_action_options(const int options,
                          const int agentID)
  {
    set_action_options(std::vector<int>(1, options), agentID);
  }

  void set_action_options(const std::vector<int> options,
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

  void set_num_agents(int _nAgents)
  {
    ENV.nAgentsPerEnvironment = _nAgents;
  }

  void disableDataTrackingForAgents(int agentStart, int agentEnd)
  {
    ENV.bTrainFromAgentData.resize(ENV.nAgentsPerEnvironment, true);
    for(int i=agentStart; i<agentEnd; i++)
      ENV.bTrainFromAgentData[i] = false;
  }
};
