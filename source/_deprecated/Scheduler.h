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

class Master
{
private:
  const Settings& settings;
  Communicator_internal* const comm;
  const std::vector<Learner*> learners;
  const Environment* const env;

  TaskQueue tasks;

  const ActionInfo& aI = env->aI;
  const StateInfo&  sI = env->sI;
  const std::vector<Agent*>& agents = env->agents;
  const int nPerRank = env->nAgentsPerRank;
  const int bTrain = settings.bTrain;
  const int nWorkers_own = settings.nWorkers_own;
  const int nThreads = settings.nThreads;
  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;
  const Uint totNumSteps = settings.totNumSteps;

  Profiler* profiler     = nullptr;

  mutable std::mutex dump_mutex;

  std::atomic<Uint> bExit {0};
  std::vector<std::thread> worker_replies;

  inline Learner* pickLearner(const Uint agentId, const Uint recvId)
  {
    assert(recvId < (Uint)nPerRank);
    if(learners.size()>1) assert(learners.size() == (Uint)nPerRank);

    assert( learners.size() == 1 || recvId < learners.size() );
    return learners.size()>1? learners[recvId] : learners[0];
  }

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
private:
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


class Master : public Worker
{
  MPI_Comm learners_data_sharing_comm = MPI_COMM_SELF;

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
