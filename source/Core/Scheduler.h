//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "../Communicators/Communicator_internal.h"
#include <thread>
#include <mutex>
#include "../Learners/Learner.h"

class Worker
{
  const Settings& settings;

  const Communicator_internal COMM;
  const MPI_Comm master_workers_comm;

  const std::vector<std::unique_ptr<Learner>> learners;
  const MPI_Comm learners_train_comm;

  const Environment& ENV = COMM.ENV;
  const std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;

  const Uint nCallingEnvironments;
};

class Worker : public Worker
{
  const Settings& settings;

  const std::unique_ptr<Communicator_internal> COMM;
  const MPI_Comm master_workers_comm;

  const std::vector<std::unique_ptr<Learner>> learners;
  const MPI_Comm learners_train_comm;

  const Environment& ENV = COMM.ENV;
  const std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;

  const int nAgentsPerRank = ENV.nAgentsPerEnvironment;
  const int bTrain = settings.bTrain;
};

class Worker
{
  const Environment& ENV = COMM.ENV;
  const std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;

  const MPI_Comm learners_train_comm;
  const std::vector<std::unique_ptr<Learner>> learners;

  Uint getLearnerID(const Uint agentIDlocal) const
  { // some asserts:
    // 1) agentID within environment must match what we know about environment
    // 2) either only one learner or ID of agent in ENV must match a learner
    // 3) if i have more than one learner, then i have one per agent in env
    assert(agentIDlocal < ENV.nAgentsPerEnvironment);
    assert(learners.size() == 1 || agentIDlocal < learners.size());
    if(learners.size()>1)
      assert(learners.size() == (size_t) ENV.nAgentsPerEnvironment);
    // if one learner, return learnerID=0, else learnID == ID of agent in ENV
    return learners.size()>1? agentIDlocal : 0;
  }

  const int bTrain = settings.bTrain;
};

class Master : public Worker
{
  MPI_Comm learners_data_sharing_comm = MPI_COMM_SELF;

  TaskQueue tasks;
  const int nWorkers_own = settings.nWorkers_own;
  const int nThreads = settings.nThreads;
  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;
  const Uint totNumSteps = settings.totNumSteps;

  Profiler* profiler     = nullptr;

  mutable std::mutex dump_mutex;

  std::atomic<Uint> bExit {0};
  std::vector<std::thread> worker_replies;

  bool learnersLockQueue() const;

  void dumpCumulativeReward(const int agent, const int worker,
    const unsigned giter, const unsigned tstep) const;

  void processWorker(const std::vector<int> workers);
  void processAgent(const int worker);

public:
  Master(Communicator_internal* const _c, const std::vector<Learner*> _l,
    Environment*const _e, Settings&_s);
  ~Master()
  {
    bExit = 1;
    for(size_t i=0; i<worker_replies.size(); i++) worker_replies[i].join();
    for(const auto& A : agents) A->writeBuffer(learn_rank);
    _dispose_object(env);
    _dispose_object(profiler);
    for(const auto& L : learners) _dispose_object(L);
    comm->sendTerminateReq();
  }

  void run();
};

class Worker
{
  MPI_Comm learners_train_comm = MPI_COMM_SELF;
  MPI_Comm learners_data_sharing_comm = MPI_COMM_SELF;
  MPI_Comm master_workers_comm = MPI_COMM_NULL;

  Communicator_internal* const comm;
  Environment* const env;
  const bool bTrain;
  std::vector<int> status;

public:
  Worker(Communicator_internal*const c, Environment*const e, Settings& s);
  ~Worker()
  {
    _dispose_object(env);
  }
  void run();
};

/*
class Client
{
private:
  Learner* const learner;
  Communicator* const comm;
  Environment* const env;
  vector<Agent*> agents;
  const ActionInfo aI;
  const StateInfo  sI;
  vector<int> status;
  void prepareState(int& iAgent, int& istatus, Real& reward);
  void prepareAction(const int iAgent);

public:
  Client(Learner*const l,Communicator*const c,Environment*const e,Settings&s);
  ~Client()
  {
    _dispose_object(env);
    _dispose_object(learner);
  }
  void run();
};
*/

#define MPI(NAME, ...)                                   \
do {                                                     \
  int MPIERR = 0;                                        \
  if(bAsync) {                                           \
    MPIERR = MPI_ ## NAME ( __VA_ARGS__ );               \
  } else {                                               \
    std::lock_guard<std::mutex> lock(mpi_mutex);         \
    MPIERR = MPI_ ## NAME ( __VA_ARGS__ );               \
  }                                                      \
  if(MPIERR not_eq MPI_SUCCESS) {                        \
    _warn("%s %d", #NAME, MPIERR);                       \
    throw std::runtime_error("MPI ERROR");               \
  }                                                      \
} while(0)
