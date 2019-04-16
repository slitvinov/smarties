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

class WorkerSockets : Master<WorkerSockets>
{
  mutable std::vector<SOCKET_REQ> reqs = std::vector<SOCKET_REQ>(nCallingEnvs+1);

protected:

  void  Recv(void* const buffer, const Uint size, const int worker) const {
    SOCKET_Brecv(buffer, size, getSocketID(worker));
  }
  void  RecvState(const int worker) const {
    const COMM_buffer& buffer = getCommBuffer(worker);
    Recv(buffer.dataStateMsg, buffer.sizeStateMsg, worker);
  }

  void Irecv(void* const buffer, const Uint size, const int worker) const {
    SOCKET_Irecv(buffer, size, getSocketID(worker), reqs[worker]);
  }
  void IrecvState(const int worker) const {
    const COMM_buffer& buffer = getCommBuffer(worker);
    Irecv(buffer.dataStateMsg, buffer.sizeStateMsg, worker);
  }

  void  Send(void* const buffer, const Uint size, const int worker) const {
    SOCKET_Bsend(buffer, size, getSocketID(worker));
  }
  void  SendAction(const int worker) const {
    const COMM_buffer& buffer = getCommBuffer(worker);
    Send(buffer.dataActionMsg, buffer.sizeActionMsg, worker);
  }

  void Isend(void* const buffer, const Uint size, const int worker) const {
    SOCKET_Isend(buffer, size, getSocketID(worker), reqs[worker]);
  }
  void IsendAction(const int worker) const {
    const COMM_buffer& buffer = getCommBuffer(worker);
    Isend(buffer.dataActionMsg, buffer.sizeActionMsg, worker);
  }

  int TestComm(const int worker) const {
    SOCKET_Test(reqs[worker].completed, reqs[worker]);
    return reqs[worker].completed;
  }
  void WaitComm(const int worker) const {
    SOCKET_Wait(reqs[worker]);
  }
};

class WorkerMPI : Master<WorkerMPI>
{
  mutable std::vector<MPI_Request> reqs =
    std::vector<MPI_Request>(nCallingEnvs+1, MPI_REQUEST_NULL);

protected:

  void  Recv(void* const buffer, const Uint size, const int worker) const {
    MPI(Recv, buffer, size, MPI_BYTE, worker, 22846,
        master_workers_comm, MPI_STATUS_IGNORE);
  }
  void  RecvState(const int worker) const {
    const COMM_buffer& buffer = getCommBuffer(worker);
    Recv(buffer.dataStateMsg, buffer.sizeStateMsg, worker);
  }

  void Irecv(void* const buffer, const Uint size, const int worker) const {
    MPI(Irecv, buffer, size, MPI_BYTE, worker, 22846,
        master_workers_comm, & reqs[worker]);
  }
  void IrecvState(const int worker) const {
    const COMM_buffer& buffer = getCommBuffer(worker);
    Irecv(buffer.dataStateMsg, buffer.sizeStateMsg, worker);
  }

  void  Send(void* const buffer, const Uint size, const int worker) const {
    MPI(Send, buffer, size, MPI_BYTE, worker, 22846, master_workers_comm);
  }
  void  SendAction(const int worker) const {
    const COMM_buffer& buffer = getCommBuffer(worker);
    Send(buffer.dataActionMsg, buffer.sizeActionMsg, worker);
  }

  void Isend(void* const buffer, const Uint size, const int worker) const {
    MPI(Isend, buffer, size, MPI_BYTE, worker, 22846,
        master_workers_comm, & reqs[worker]);
  }
  void IsendAction(const int worker) const {
    const COMM_buffer& buffer = getCommBuffer(worker);
    Isend(buffer.dataActionMsg, buffer.sizeActionMsg, worker);
    MPI_Request_free(& );
  }

  int TestComm(const int worker) const {
    int completed = 0; MPI_Status mpistatus;
    MPI(Test, &reqs[worker], &completed, &mpistatus);
    assert(worker == mpistatus.MPI_SOURCE);
    return completed;
  }
  void WaitComm(const int worker) const {
    MPI_Status mpistatus;
    MPI(Wait, &reqs[worker], &mpistatus);
    assert(worker == mpistatus.MPI_SOURCE);
  }
};

template <typename CommType>
class Master : public Worker
{
protected:

  CommType * interface() { return static_cast<CommType*> (this); }
};

class Worker
{
protected:
  const Settings& settings;
  TaskQueue tasks;

  const Communicator_internal COMM;
  const MPI_Comm master_workers_comm;
  const MPI_Comm workerless_masters_comm;

  const std::vector<std::unique_ptr<Learner>> learners;
  const MPI_Comm learners_train_comm;

  const Environment& ENV = COMM.ENV;
  const std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;

  const Uint nCallingEnvs;

  // small utility functions:
  Uint getLearnerID(const Uint agentIDlocal) const;
  bool learnersBlockingDataAcquisition() const;
  void dumpCumulativeReward(const Agent&, const Uint k, const Uint t) const;

  void answerStateActionCaller(const int bufferID);

  const int bTrain = settings.bTrain;


  int getSocketID(const Uint worker) const {
    assert( worker>=0 && worker <= COMM.SOCK.clients.size() );
    return worker>0? COMM.SOCK.clients[worker-1] : COMM.SOCK.server;
  }
  const COMM_buffer& buffer getCommBuffer(const Uint worker) const {
    assert( worker>0 && worker <= COMM.BUFF.size() );
    return * COMM.BUFF[worker-1].get();
  }
};

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
