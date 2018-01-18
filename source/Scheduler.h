//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#pragma once
#include "Communicator.h"
#include <thread>
#include <mutex>
#include "Learners/Learner.h"

enum IrecvStatus { OPEN, DOING, EMPTY };

class Master
{
private:
  const MPI_Comm slavesComm;
  const vector<Learner*> learners;
  Environment* const env;
  const ActionInfo aI;
  const StateInfo  sI;
  const vector<Agent*> agents;
  const int bTrain, nPerRank, saveFreq, nSlaves, nThreads, learn_rank, learn_size, totNumSteps, outSize, inSize;
  const vector<double*> inpBufs;
  const vector<double*> outBufs;
  long int stepNum = 0;
  vector<IrecvStatus> slaveIrecvStatus;
  vector<Uint> agentSortingCheck;
  vector<MPI_Request> requests;
  vector<int> postponed_queue;
  Profiler* profiler     = nullptr;
  Profiler* profiler_int = nullptr;
  std::mutex mpi_mutex, dump_mutex;

  int& nTasks;

  inline void prepareLearners()
  {
    for(const auto& L : learners)
      L->prepareData(); //sync data, make sure we can sample

    //this is the last possible time to finish the blocking mpi MPI_Allreduce
    // and finally perform the actual gradient step
    for(const auto& L : learners)
      L->synchronizeGradients();
  }

  inline bool learnersLockQueue()
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
    //    on a slave, all other agents on that slave must send their term state
    //    before sending any new initial state

    // However, no learner can stop others from getting data (vector of algos)
    bool locked = true;
    for(const auto& L : learners)
      locked = locked && not L->unlockQueue(); // if any wants to unlock...

    return locked;
  }

  inline bool learnersGradientReady()
  {
    // can we perform any batch gradient update?
     bool ready = false;
     for(const auto& L : learners)
       ready = ready || L->batchGradientReady();
     return ready;
  }

  static inline vector<double*> alloc_bufs(const int size, const int num)
  {
    vector<double*> ret(num, nullptr);
    for(int i=0; i<num; i++) ret[i] = _alloc(size);
    return ret;
  }

  inline void processRequest(const int slave);

  inline void sendBuffer(const int i)
  {
    assert(i>0);
    MPI_Request tmp;
    #pragma omp critical
    MPI_Isend(outBufs[i-1], outSize, MPI_BYTE, i, 0, slavesComm, &tmp);
    MPI_Request_free(&tmp); //Not my problem
    debugS("Sent action to slave %d: [%s]", i,
      print(vector<Real>(outBufs[i-1], outBufs[i-1]+aI.dim)).c_str());
  }

  inline void recvBuffer(const int i)
  {
    #pragma omp critical
    MPI_Irecv(inpBufs[i-1], inSize, MPI_BYTE, i, 1, slavesComm, &requests[i-1]);
    slaveIrecvStatus[i-1] = OPEN;
  }

  inline int readNTasks() const
  {
    return nTasks;
  }
  inline void addToNTasks(const int add) const
  {
    #pragma omp atomic
    nTasks += add;
  }

  inline void spawnTrainingTasks() const
  {
    for(const auto& L : learners)
      L->spawnTrainTasks();
  }

public:
  Master(MPI_Comm _c,const vector<Learner*>_l,Environment*const _e,Settings&_s);
  ~Master()
  {
    _dispose_object(env);
    for(int i=0; i<nSlaves; i++) _dealloc(inpBufs[i]);
    for(int i=0; i<nSlaves; i++) _dealloc(outBufs[i]);
    for(const auto& L : learners) _dispose_object(L);
  }

  void sendTerminateReq()
  {
    //it's awfully ugly, i send -256 to kill the slaves... but...
    //what are the chances that learner sends action -256.(+/- eps) to clients?
    printf("nslaves %d\n",nSlaves);
    for (int slave=1; slave<=nSlaves; slave++) {
      outBufs[slave-1][0] = _AGENT_KILLSIGNAL;
      #pragma omp critical
      MPI_Ssend(outBufs[slave-1], outSize, MPI_BYTE, slave, 0, slavesComm);
    }
  }

  int run();
};

class Slave
{
private:
  Communicator* const comm;
  Environment* const env;
  const bool bTrain;
  vector<int> status;

public:
  Slave(Communicator*const c, Environment*const e, Settings& s);
  ~Slave()
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
