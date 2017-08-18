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

enum IrecvStatus { OPEN, DOING, SEND, OVER, STALL, EMPTY };

class Master
{
private:
  const MPI_Comm slavesComm;
  Learner* const learner;
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
  vector<pair<int,int>> postponed_queue;
  Profiler* profiler;
  std::mutex mpi_mutex;
  //to check compatiblity with on-policy learning:
  int learnerReadyForAgent(const int slave, const int agent) const;
  int recvState(const int slave);
  //void sendAction(const int slave, const int iAgent);

  static inline vector<double*> alloc_bufs(const int size, const int num)
  {
    vector<double*> ret(size, nullptr);
    for(int i=0; i<num; i++) ret[i] = _alloc(size);
    return ret;
  }

  //mutable std::mutex client_mutex;
  //Real avgNbusy = nSlaves;
  //int nServing = 0;
  void processRequest(const int slave, const int agent);

  //inline int readNServing() const
  //{
  //  lock_guard<mutex> lock(client_mutex);
  //  return nServing;
  //}
  inline void addToNTasks(const int add) const
  {
    learner->addToNTasks(add);
    //lock_guard<mutex> lock(client_mutex);
    //nServing += add;
  }
  inline void spawnTrainingTasks(bool& first, const bool slaves_waiting) const
  {
    #ifndef FULLTASKING
    learner->spawnTrainTasks(slaves_waiting); //spawn all tasks
    #else
      #if 0
      const int nServerThreads = readNServing();
      const int nTrainTasks = learner->readNTasks() - nServerThreads;
      nSlaves tasks are reserved to handle slaves, if comm queue is empty
      if( !first && !postponed_queue.size() )
        avgNbusy += 1e-6 * (nServerThreads-avgNbusy);
      first = false;
      const int goodNreserved = max((int) ceil(avgNbusy), 1);
      const int nReservedTasks = postponed_queue.size()? 0 : goodNreserved;
      const int availTasks = nThreads - nTrainTasks - nReservedTasks;
      learner->spawnTrainTasks(availTasks);
      #else
      const int nReservedTasks = postponed_queue.size()? 0 : 1;
      learner->spawnTrainTasks(nThreads - learner->readNTasks() - nReservedTasks);
      #endif
    #endif
  }

public:
  Master(MPI_Comm comm, Learner*const learner, Environment*const env, Settings& settings);
  ~Master()
  {
    _dispose_object(env);
    for(int i=0; i<nSlaves; i++) _dealloc(inpBufs[i]);
    for(int i=0; i<nSlaves; i++) _dealloc(outBufs[i]);
    _dispose_object(learner);
  }

  void sendTerminateReq(const double msg = -256)
  {
    //it's awfully ugly, i send -256 to kill the slaves... but...
    //what are the chances that learner sends action -256.(+/- eps) to clients?
    printf("nslaves %d\n",nSlaves);
    for (int slave=1; slave<=nSlaves; slave++) {
      outBufs[slave-1][0] =  msg;
      MPI_Ssend(outBufs[slave-1], outSize, MPI_BYTE, slave, 0, slavesComm);
    }
  }

  int run();

  void save();
  void restart(string fname);
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
