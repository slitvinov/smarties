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

class Master
{
private:
  const MPI_Comm slavesComm;
  const vector<Learner*> learners;
  Environment* const env;
  const ActionInfo aI;
  const StateInfo  sI;
  const vector<Agent*> agents;
  const int bTrain, nPerRank, nSlaves, nThreads, learn_rank, learn_size, totNumSteps, outSize, inSize;
  const vector<double*> inpBufs;
  const vector<double*> outBufs;
  mutable long int stepNum = 0, iternum = 0;
  mutable vector<MPI_Request> requests;
  Profiler* profiler     = nullptr;
  Profiler* profiler_int = nullptr;
  mutable std::mutex mpi_mutex;
  mutable std::mutex dump_mutex;
  mutable std::ostringstream rewardsBuffer;

  inline Uint readTimeSteps() const
  {
    Uint ret;
    #pragma omp atomic read
    ret = stepNum;
    return ret;
  }

  inline void prepareLearners() const
  {
    for(const auto& L : learners)
      L->prepareData(); //sync data, make sure we can sample

    //this is the last possible time to finish the blocking mpi MPI_Allreduce
    // and finally perform the actual gradient step
    for(const auto& L : learners)
      L->synchronizeGradients();
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
    //    on a slave, all other agents on that slave must send their term state
    //    before sending any new initial state

    // However, no learner can stop others from getting data (vector of algos)
    bool locked = true;
    for(const auto& L : learners)
      locked = locked && L->lockQueue(); // if any wants to unlock...

    return locked;
  }

  void flushRewardBuffer()
  {
    streampos pos = rewardsBuffer.tellp(); // store current location
    rewardsBuffer.seekp(0, ios_base::end); // go to end
    bool empty = rewardsBuffer.tellp()==0; // check size == 0 ?
    rewardsBuffer.seekp(pos);              // restore location
    if(empty) return;                      // else update rewards log
    char path[256];
    sprintf(path, "cumulative_rewards_rank%02d.dat", learn_rank);
    ofstream outf(path, ios::app);
    outf << rewardsBuffer.str();
    rewardsBuffer.str(std::string());      // empty buffer
    outf.flush();
    outf.close();
  }

  inline void dumpCumulativeReward(const int agent, const unsigned iter,
    const unsigned tstep) const
  {
    if (iter == 0 && bTrain) return;

    lock_guard<mutex> lock(dump_mutex);
    rewardsBuffer<<iter<<" "<<tstep<<" "<<agent<<" "
    <<agents[agent]->transitionID<<" "<<agents[agent]->cumulative_rewards<<endl;
    rewardsBuffer.flush();
  }

  static inline vector<double*> alloc_bufs(const int size, const int num)
  {
    vector<double*> ret(num, nullptr);
    for(int i=0; i<num; i++) ret[i] = _alloc(size);
    return ret;
  }

  void processSlave(const int slave);
  void processAgent(const int slave, const MPI_Status mpistatus);

  inline void sendBuffer(const int slave, const int agent)
  {
    assert(slave>0 && slave <= (int) outBufs.size());
    for(Uint i=0; i<aI.dim; i++)
      outBufs[slave-1][i] = agents[agent]->a->vals[i];

    debugS("Sent action to slave %d: [%s]", slave,
      print(Rvec(outBufs[slave-1], outBufs[slave-1]+aI.dim)).c_str());
    MPI_Request tmp;
    lock_guard<mutex> lock(mpi_mutex);
    MPI_Isend(outBufs[slave-1], outSize, MPI_BYTE, slave, 0, slavesComm, &tmp);
    MPI_Request_free(&tmp); //Not my problem
  }

  inline void recvBuffer(const int i)
  {
    lock_guard<mutex> lock(mpi_mutex);
    MPI_Irecv(inpBufs[i-1], inSize, MPI_BYTE, i, 1, slavesComm, &requests[i-1]);
  }

  inline void spawnTrainingTasks_par() const
  {
    for(const auto& L : learners)
      L->spawnTrainTasks_par();
  }

  inline void spawnTrainingTasks_seq() const
  {
    for(const auto& L : learners)
      L->spawnTrainTasks_seq();
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
      lock_guard<mutex> lock(mpi_mutex);
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
