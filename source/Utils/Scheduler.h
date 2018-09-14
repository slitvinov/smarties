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
  const MPI_Comm workersComm;
  const vector<Learner*> learners;
  Environment* const env;
  const ActionInfo& aI = env->aI;
  const StateInfo&  sI = env->sI;
  const vector<Agent*>& agents = env->agents;
  const int nPerRank = env->nAgentsPerRank, bTrain, nWorkers, nThreads;
  const int learn_rank, learn_size;
  const Uint totNumSteps, bAsync;
  const Uint outSize = env->aI.dim * sizeof(double);
  const Uint inSize = (3 + env->sI.dim) * sizeof(double);
  const vector<double*> inpBufs = alloc_bufs(inSize, nWorkers);
  const vector<double*> outBufs = alloc_bufs(outSize,nWorkers);
  Uint iterNum = 0; // no need to restart this one

  bool bNeedSequentialTasks = false;
  mutable vector<MPI_Request> requests = vector<MPI_Request>(nWorkers, MPI_REQUEST_NULL);
  Profiler* profiler     = nullptr;
  mutable mutex mpi_mutex;
  mutable mutex dump_mutex;
  mutable vector<ostringstream> rewardsBuffer = vector<ostringstream>(nPerRank);

  std::atomic<Uint> bExit {0};
  vector<std::thread> worker_replies;

  inline Uint getMinSeqId() const {
    Uint lowest = learners[0]->nSeqsEval();
    for(size_t i=1; i<learners.size(); i++) {
      const Uint tmp = learners[i]->nSeqsEval();
      if(tmp<lowest) lowest = tmp;
    }
    // if agents share learning algo, return number of eps performed by env:
    if(learners.size() == 1) lowest /= nPerRank;
    return lowest;
  }
  inline Uint getMinStepId() const {
    Uint lowest = learners[0]->tStepsTrain();
    for(size_t i=1; i<learners.size(); i++) {
      const Uint tmp = learners[i]->tStepsTrain();
      if(tmp<lowest) lowest = tmp;
    }
    // if agents share learning algo, return number of turns performed by env:
    if(learners.size() == 1) lowest /= nPerRank;
    return lowest;
  }

  inline Learner* pickLearner(const Uint agentId, const Uint recvId)
  {
    assert(agentId<agents.size() && recvId<(Uint)nPerRank);
    if(learners.size()>1) assert(learners.size() == (Uint)nPerRank);

    assert( learners.size() == 1 || recvId < learners.size() );
    return learners.size()>1? learners[recvId] : learners[0];
  }

  inline bool learnersLockQueue() const
  {
    //When would a learning algo stop acquiring more data?
    //Off Policy algos:
    // - User specifies a ratio of observed trajectories to gradient steps.
    //    Comm is restarted or paused to maintain this ratio consant.
    //On Policy algos:
    // - if collected enough trajectories for current batch, then comm is paused
    //    untill gradient is applied (or nepocs are done), then comm restarts
    //    to obtain fresh on policy samples
    // Note:
    // - on policy traj. storage assumes that when agent reaches terminal state
    //    on a worker, all other agents on that worker must send their term state
    //    before sending any new initial state

    // However, no learner can stop others from getting data (vector of algos)
    bool locked = true;
    for(const auto& L : learners)
      locked = locked && L->lockQueue(); // if any wants to unlock...

    return locked;
  }

  void flushRewardBuffer()
  {
    for(int i=0; i<nPerRank; i++)
    {
      ostringstream& agentBuf = rewardsBuffer[i];
      streampos pos = agentBuf.tellp(); // store current location
      agentBuf.seekp(0, ios_base::end); // go to end
      bool empty = agentBuf.tellp()==0; // check size == 0 ?
      agentBuf.seekp(pos);              // restore location
      if(empty) continue;               // else update rewards log
      char path[256];
      sprintf(path, "agent_%02d_rank%02d_cumulative_rewards.dat", i,learn_rank);
      ofstream outf(path, ios::app);
      outf << agentBuf.str();
      agentBuf.str(std::string());      // empty buffer
      outf.flush();
      outf.close();
    }
  }

  inline void dumpCumulativeReward(const int agent, const int worker,
    const unsigned giter, const unsigned tstep) const
  {
    if (giter == 0 && bTrain) return;

    const int ID = (worker-1) * nPerRank + agent;
    lock_guard<mutex> lock(dump_mutex);
    rewardsBuffer[agent]<<giter<<" "<<tstep<<" "<<worker<<" "
      <<agents[ID]->transitionID<<" "<<agents[ID]->cumulative_rewards<<endl;
    rewardsBuffer[agent].flush();
  }

  static inline vector<double*> alloc_bufs(const int size, const int num)
  {
    vector<double*> ret(num, nullptr);
    for(int i=0; i<num; i++) ret[i] = _alloc(size);
    return ret;
  }
  static inline vector<atomic<Uint>*> alloc_counts(const int num)
  {
    vector<atomic<Uint>*> ret(num, nullptr);
    for(int i=0; i<num; i++) ret[i] = new atomic<Uint>{0};
    return ret;
  }

  void processWorker(const int worker);
  void processAgent(const int worker, const MPI_Status mpistatus);

  inline void sendBuffer(const int i, const int agent)
  {
    assert(i>0 && i <= (int) outBufs.size());
    if(agents[agent]->Status < TERM_COMM) agents[agent]->copyAct(outBufs[i-1]);
    else intToDoublePtr( getMinStepId(), outBufs[i-1] );

    debugS("Sent action to worker %d: [%s]", i,
      print(Rvec(outBufs[i-1], outBufs[i-1]+aI.dim)).c_str());
    MPI_Request tmp;
    if(bAsync) { // MPI impl allows maximum thread safety
      MPI_Isend(outBufs[i-1],outSize, MPI_BYTE, i,0,workersComm,&tmp);
    } else {
      lock_guard<mutex> lock(mpi_mutex);
      MPI_Isend(outBufs[i-1],outSize, MPI_BYTE, i,0,workersComm,&tmp);
    }
    MPI_Request_free(&tmp); //Not my problem
  }

  inline void recvBuffer(const int i)
  {
    if(bAsync) { // MPI impl allows maximum thread safety
      MPI_Irecv(inpBufs[i-1], inSize, MPI_BYTE, i,1,workersComm,&requests[i-1]);
    } else {
      lock_guard<mutex> lock(mpi_mutex);
      MPI_Irecv(inpBufs[i-1], inSize, MPI_BYTE, i,1,workersComm,&requests[i-1]);
    }
  }

  inline int testBuffer(const int i, MPI_Status& mpistatus)
  {
    int completed = 0;
    if(bAsync) { // MPI impl allows maximum thread safety
      MPI_Test(&requests[i-1], &completed, &mpistatus);
    } else {
      lock_guard<mutex> lock(mpi_mutex);
      MPI_Test(&requests[i-1], &completed, &mpistatus);
    }
    return completed;
  }

public:
  Master(MPI_Comm _c,const vector<Learner*>_l,Environment*const _e,Settings&_s);
  ~Master()
  {
    bExit = 1;
    for(int i=0; i<nWorkers; i++) worker_replies[i].join();
    for(const auto& A : agents) A->writeBuffer(learn_rank);
    _dispose_object(env);
    _dispose_object(profiler);
    for(int i=0; i<nWorkers; i++) _dealloc(inpBufs[i]);
    for(int i=0; i<nWorkers; i++) _dealloc(outBufs[i]);
    for(const auto& L : learners) _dispose_object(L);
    flushRewardBuffer();
  }

  void sendTerminateReq()
  {
    //it's awfully ugly, i send -256 to kill the workers... but...
    //what are the chances that learner sends action -256.(+/- eps) to clients?
    for (int worker=1; worker<=nWorkers; worker++) {
      outBufs[worker-1][0] = AGENT_KILLSIGNAL;
      MPI_Ssend(outBufs[worker-1], outSize, MPI_BYTE, worker, 0, workersComm);
    }
  }

  void run();
};

class Worker
{
private:
  Communicator_internal* const comm;
  Environment* const env;
  const bool bTrain;
  vector<int> status;

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
