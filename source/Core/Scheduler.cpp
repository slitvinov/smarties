//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Scheduler.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

Master::Master(Communicator_internal* const _c, const std::vector<Learner*> _l,
  Environment*const _e, Settings&_s): settings(_s),comm(_c),learners(_l),env(_e)
{
  if(nWorkers_own*nPerRank != static_cast<int>(agents.size()))
    die("Mismatch in master's nWorkers nPerRank nAgents.");

  worker_replies.reserve(nWorkers_own);
  //the following Irecv will be sent after sending the action
  for(int i=1; i<=nWorkers_own; i++) comm->recvBuffer(i);

  for(const auto& L : learners) L->setupTasks(tasks);
}

void Master::run()
{
  {
    #pragma omp parallel num_threads(nThreads)
    {
      std::vector<int> shareWorkers;
      const int thrID = omp_get_thread_num(), thrN = omp_get_num_threads();
      for(int i=1; i<=nWorkers_own; i++)
       if( thrID == (( ( (-i)%thrN ) +thrN ) %thrN) ) shareWorkers.push_back(i);

      #pragma omp critical
      if(shareWorkers.size()) worker_replies.push_back(
        std::thread( [&, shareWorkers] () { processWorker(shareWorkers); }));
    }
  }

  Uint minNdataB4Train = learners[0]->nObsB4StartTraining;
  int firstLearnerStart = 0, isStarted = 0, percentageReady = -5;
  for(size_t i=1; i<learners.size(); i++) {
    Uint tmp = learners[i]->nObsB4StartTraining;
    if(tmp<minNdataB4Train) {
      minNdataB4Train = tmp;
      firstLearnerStart = i;
    }
  }
  const auto isTrainingStarted = [&]() {
    if(not isStarted && learn_rank==0) {
      const auto nCollected = learners[firstLearnerStart]->locDataSetSize();
      const int currPerc = nCollected * 100./(Real) minNdataB4Train;
      if(nCollected >= minNdataB4Train) isStarted = true;
      else if(currPerc > percentageReady+5) {
       percentageReady = currPerc;
       printf("\rCollected %d%% of data required to begin training. ",currPerc);
       fflush(0); //otherwise no show on some platforms
      }
    }
  };
  const auto isTrainingOver = [&](const Learner* const L) {
    // if agents share learning algo, return number of turns performed by env
    // instead of sum of timesteps performed by each agent
    const Real factor = learners.size() == 1? 1.0/nPerRank : 1;
    const long dataCounter = bTrain? L->nLocTimeStepsTrain() : L->nSeqsEval();
    return dataCounter * factor >= totNumSteps;
  };

  while(1)
  {
    isTrainingStarted();

    tasks.run();

    bool over = true;
    for(const auto& L : learners) over = over && isTrainingOver(L);
    if (over) break;
  }
}

void Master::processWorker(const std::vector<int> workers)
{
  while(1)
  {
    for( const int worker : workers )
    {
      assert(worker>0 && worker <= (int) nWorkers_own);
      int completed = comm->testBuffer(worker);

      // Learners lock workers queue if they have enough data to advance step
      while ( bTrain && completed && learnersLockQueue() ) {
        usleep(1);
        if( bExit.load() > 0 ) break;
      }

      if(completed) processAgent(worker);

      usleep(1);
    }
  }
}

void Master::processAgent(const int worker)
{
  //read from worker's buffer:
  std::vector<double> recv_state(sI.dim);
  int recv_agent  = -1; // id of agent inside environment
  int recv_status = -1; // initial/normal/termination/truncation of episode
  double reward   =  0;
  comm->unpackState(worker-1, recv_agent, recv_status, recv_state, reward);
  if (recv_status == FAIL_COMM) die("app crashed");

  const int agent = (worker-1) * nPerRank + recv_agent;
  Learner*const learner = pickLearner(agent, recv_agent);

  agents[agent]->update(recv_status, recv_state, reward);
  //pick next action and ...do a bunch of other stuff with the data:
  learner->select(*agents[agent]);

  debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]",
    agent, agents[agent]->Status, agents[agent]->sOld._print().c_str(),
    agents[agent]->s._print().c_str(), agents[agent]->r,
    agents[agent]->a._print().c_str());

  std::vector<double> actVec = agents[agent]->getAct();
  if(agents[agent]->Status >= TERM_COMM) {
    const Real factor = learners.size() == 1? 1.0/nPerRank : 1;
    const auto nSteps = learner->nLocTimeStepsTrain();
    actVec[0] = factor * nSteps;
    dumpCumulativeReward(recv_agent, worker, learner->nGradSteps(), nSteps);
  }

  debugS("Sent action to worker %d: [%s]", worker, print(actVec).c_str() );
  comm->sendBuffer(worker, actVec);
  comm->recvBuffer(worker); // prepare next recv
}

bool Master::learnersLockQueue() const
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
    locked = locked && L->blockDataAcquisition(); // if any wants to unlock...

  return locked;
}

Worker::Worker(Communicator_internal*const _c,Environment*const _e,Settings&_s)
: comm(_c), env(_e) {}

void Worker::run()
{
  while(true) {

    while(true) {
      if (comm->recvStateFromApp()) break; //sim crashed

      if (comm->sendActionToApp() ) {
        die("Worker exiting");
        return;
      }
    }
    die("Simulation crash");
    //if we are training, then launch again, otherwise exit
    //if (!bTrain) return;
    comm->launch();
  }
}

void Master::dumpCumulativeReward(const int agent, const int worker,
  const unsigned giter, const unsigned tstep) const
{
  if (giter == 0 && bTrain) return;

  const int ID = (worker-1) * nPerRank + agent;
  char path[256];
  sprintf(path, "agent_%02d_rank%02d_cumulative_rewards.dat", agent,learn_rank);

  std::lock_guard<std::mutex> lock(dump_mutex);
  FILE * pFile = fopen (path, "a");
  fprintf (pFile, "%u %u %d %d %f\n", giter, tstep, worker,
    agents[ID]->transitionID, agents[ID]->cumulative_rewards);
  fflush (pFile);
  fclose (pFile);
}

// MPI COMMUNICATORS:
/*
  - masters_train_comm : all masters
  - masters_data_sharing_comm = for workerless masters
  - master_workers_comm
  - workers_application_comm
*/

void Master::synchronizeEnvironments()
{
  std::function<void(void*, size_t)> recvBuffer;

  if( COMM->clientSockets.size() > 0 )
  { // master with apps connected through sockets (on the same compute node)
    assert( MPICommSize(master_workers_comm) <= 1 );
    recvBuffer = [&](void* buffer, size_t size)
    {
      const size_t numSockets = COMM->clientSockets.size();
      const int err = SOCKET_Recv(buffer, size, COMM->clientSockets[0]);
      if(err not_eq 0) die("SOCKET hung up communication");

      for(size_t i=1; i < numSockets; ++i) {
        //std::unique_ptr<void> testbuffer = std::make_unique<void> // TODO
        const int err = SOCKET_Recv(buffer, size, COMM->clientSockets[i]);
        // TODO check against testbuffer
      }

      assert( MPICommRank(masters_data_sharing_comm) == 0 );
      for(int i=1; i<MPICommRank(masters_data_sharing_comm); ++i)
        MPI_Send(buffer, size, MPI_BYTE, i, 368637, masters_data_sharing_comm);
    };
  }
  else if ( MPICommSize(master_workers_comm) > 1 )
  { // master with apps running through mpi workers (different compute nodes)
    assert( MPICommRank(master_workers_comm) == 0 );
    recvBuffer = [&](void* buffer, size_t size)
    {
      MPI_Recv(buffer, size, MPI_BYTE, 1, 368637 /* ENVMDP */,
          master_workers_comm, MPI_STATUS_IGNORE);
      // size of comm is number of workers plus master:
      for(size_t i=2; i < MPICommSize(master_workers_comm); ++i) {
        //std::unique_ptr<void> testbuffer = std::make_unique<void> // TODO
        MPI_Recv(buffer, size, MPI_BYTE, i, 368637 /* ENVMDP */,
            master_workers_comm, MPI_STATUS_IGNORE);
        // TODO check against testbuffer
      }

      assert( MPICommRank(masters_data_sharing_comm) == 0 );
      for(int i=1; i<MPICommRank(masters_data_sharing_comm); ++i)
        MPI_Send(buffer, size, MPI_BYTE, i, 368637, masters_data_sharing_comm);
    };
  }
  else if ( MPICommRank(masters_data_sharing_comm) > 0 )
  { // master without own workers : receive data from an other master
    recvBuffer = [&](void* buffer, size_t size)
    {
      MPI_Recv(buffer, size, MPI_BYTE, 0, 368637 /* ENVMDP */,
          masters_data_sharing_comm, MPI_STATUS_IGNORE);
    }
  }
  else die("master rank with no means to get environment description");

  synchronizeEnvironments( recvBuffer );
}

void Worker::synchronizeEnvironments()
{
  if( MPICommSize(master_workers_comm) <= 1 )
    die("worker has no mpi connection with master");

  std::function<void(void*, size_t)> recvBuffer;
  // types of worker: either has sockets or it is the application
  if( COMM->clientSockets.size() > 0 )
  { // worker with apps connected through sockets (on the same compute node)
    recvBuffer = [&](void* buffer, size_t size)
    {
      const size_t numSockets = COMM->clientSockets.size();
      const int err = SOCKET_Recv(buffer, size, COMM->clientSockets[0]);
      if(err not_eq 0) die("SOCKET hung up communication");
      for(size_t i=1; i < numSockets; ++i) {
        //std::unique_ptr<void> testbuffer = std::make_unique<void> // TODO
        const int err = SOCKET_Recv(buffer, size, COMM->clientSockets[i]);
        // TODO check against testbuffer
      }
      assert( MPICommRank(master_workers_comm) > 0 );
      MPI_Send(buffer, size, MPI_BYTE, 0, 368637, master_workers_comm);
    };
  }
  else
  {
    recvBuffer = [&](void* buffer, size_t size) {
      MPI_Send(buffer, size, MPI_BYTE, 0, 368637, master_workers_comm);
    };
  }

  synchronizeEnvironments( recvBuffer );
}

void Worker::loopSocketToMaster()
{
  auto& BUF = COMM->BUF;
  const size_t numSockets = COMM->clientSockets.size();
  std::vector<unsigned> socketRequests(numSockets, 0);
  // prepare state recv buffers for non-blocking socket communication:
  std::vector<std::vector<double>> socketBuffers(numSockets);
  for(size_t i=0; i<numSockets; ++i) socketBuffers[i] = BUF.dataStateMsg;

  while(true)
  {
    for(size_t i=0; i < numSockets; ++i)
    {
      // attempt to receive the msg through socket i
      const int error = SOCKET_Irecv(socketBuffers[i].data(), BUF.sizeStateMsg,
          COMM->clientSockets[i], & socketRequests[i]);

      if( error not_eq 0 )
      {
        _warn("lost communication with socket %d\n", i);
        socketBuffers[i][0] = 0;
        socketBuffers[i][1] = status2int(FAIL) +.1; // tell master about problem
        socketRequests[i] = BUF.sizeStateMsg; // trigger send message to master
      }

      // check if we received a message from the socket:
      if(socketRequests[i] >= BUF.sizeStateMsg)
      {
        // first copy message to default state memory buffer:
        BUF.dataStateMsg = socketBuffers[i];
        stepWorkerToMaster();
        // now the message from the master will be in default action mem buffer
        bool SIGTERM = std::fabs(BUF.dataActionMsg[0]-AGENT_KILLSIGNAL) < 2e-16;
        if(SIGTERM) return; //TODO: destructor will send all termination signals
        SOCKET_Bsend(BUF.dataActionMsg.data(), BUF.sizeActionMsg,
            COMM->clientSockets[i]);
      }
      else
      {
        usleep(1); // wait for application to send a state without burning a cpu
      }
    }
  }
}

void Master::stepMasterToWorker()
{
  assert(MPI.rank_learn_pool > 0);
  assert(MPI.comm_learn_pool != MPI_COMM_NULL);
  auto& BUF = COMM->BUF;

  assert(i>0 && i <= (int) outBufs.size() && V.size() == (size_t) nActions);
  std::copy (V.begin(), V.end(), outBufs[i-1] );

  if(bSpawnApp)
    send_all(clientSockets[i-1], outBufs[i-1], size_action);
  else {
    MPI_Request tmp;
    MPI(Isend, outBufs[i-1], size_action, MPI_BYTE, i, 0,
      comm_learn_pool, &tmp);
    MPI(Request_free, &tmp); //Not my problem
  }

  if(bSpawnApp) return;

  MPI(Irecv, inpBufs[i-1], size_state, MPI_BYTE, i, 1,
      comm_learn_pool, &requests[i-1]);

  // MPI MSG to master of a single state:
  MPI_Request send_request, recv_request;
  MPI_Isend(BUF.dataStateMsg.data(), BUF.sizeStateMsg, MPI_BYTE, 0, 1,
      MPI.comm_learn_pool, &send_request);
  MPI_Request_free(&send_request);
  // MPI MSG from master of a single action:
  MPI_Irecv(BUF.dataActionMsg.data(), BUF.sizeActionMsg, MPI_BYTE, 0, 0,
      MPI.comm_learn_pool, &recv_request);


}

int Master::testBuffer(const int i)
{
  int completed = 0;
  if(bSpawnApp)
  {
    const int bytes = recv_all(clientSockets[i-1], inpBufs[i-1], size_state);
    if(bytes not_eq size_state) {
      if (bytes == 0) _die("socket %d hung up\n", clientSockets[i-1]);
      else die("(1) recv");
      intToDoublePtr(FAIL_COMM, data_state+1);
    }
    completed = 1;
  }
  else
  {
    MPI_Status mpistatus;
    MPI(Test, &requests[i-1], &completed, &mpistatus);
    if(completed) assert(i == mpistatus.MPI_SOURCE);
  }
  return completed;
}

void Worker::stepWorkerToMaster()
{
  assert(MPI.rank_learn_pool > 0);
  assert(MPI.comm_learn_pool != MPI_COMM_NULL);
  auto& BUF = COMM->BUF;

  if(MPI.rank_inside_app <= 0 || bEnvDistributedAgents)
  {
    // MPI MSG to master of a single state:
    MPI_Request send_request, recv_request;
    MPI_Isend(BUF.dataStateMsg.data(), BUF.sizeStateMsg, MPI_BYTE, 0, 1,
        MPI.comm_learn_pool, &send_request);
    MPI_Request_free(&send_request);
    // MPI MSG from master of a single action:
    MPI_Irecv(BUF.dataActionMsg.data(), BUF.sizeActionMsg, MPI_BYTE, 0, 0,
        MPI.comm_learn_pool, &recv_request);
    while(1) // wait action from master without burning cpu resources
    {
      int completed = 0;
      MPI_Test(&recv_request, &completed, MPI_STATUS_IGNORE);
      if (completed) break;
      usleep(1);
    }

    if(not bEnvDistributedAgents && MPI.size_inside_app > 1)
    {
      //Then this is rank 0 of an environment with centralized agents.
      //Broadcast same action to members of the gang:
      MPI_Bcast(BUF.dataActionMsg.data(), BUF.sizeActionMsg, MPI_BYTE, 0,
                MPI.comm_inside_app)
    }
  }
  else // rank_inside_app>0 && not bEnvDistributedAgents
  {
    assert(MPI.size_inside_app>1);
    //Else this function was called by rank>0 of an app with centralized agents.
    //Therefore, recv the action obtained from master:
    MPI_Bcast(BUF.dataActionMsg.data(), BUF.sizeActionMsg, MPI_BYTE, 0,
              MPI.comm_inside_app)
  }

  learner_step_id = (unsigned) BUF.dataActionMsg[0];
}

/*
Client::Client(Learner*const _l, Communicator*const _c, Environment*const _e,
    Settings& _s):
    learner(_l), comm(_c), env(_e), agents(_e->agents), aI(_e->aI), sI(_e->sI),
    sOld(_e->sI), sNew(_e->sI), aOld(_e->aI, &_s.generators[0]),
    aNew(_e->aI, &_s.generators[0]), status(_e->agents.size(),1)
{}

void Client::run()
{
  vector<double> state(env->sI.dim);
  int iAgent, agentStatus;
  double reward;

  while(true)
  {
    if (comm->recvStateFromApp()) break; //sim crashed

    prepareState(iAgent, agentStatus, reward);
    learner->select(iAgent, sNew, aNew, sOld, aOld, agentStatus, reward);

    debugS("Agent %d: [%s] -> [%s] with [%s] rewarded with %f going to [%s]\n",
        iAgent, sOld._print().c_str(), sNew._print().c_str(),
        aOld._print().c_str(), reward, aNew._print().c_str());
    status[iAgent] = agentStatus;

    if(agentStatus != _AGENT_LASTCOMM) {
      prepareAction(iAgent);
      comm->sendActionToApp();
    } else {
      bool bDone = true; //did all agents reach terminal state?
      for (Uint i=0; i<status.size(); i++)
        bDone = bDone && status[i] == _AGENT_LASTCOMM;
      bDone = bDone || env->resetAll; //or does env end is any terminates?

      if(bDone) {
        comm->answerTerminateReq(-1);
        return;
      }
      else comm->answerTerminateReq(1);
    }
  }
}

void Client::prepareState(int& iAgent, int& istatus, Real& reward)
{
  Rvec recv_state(sNew.sInfo.dim);

  unpackState(comm->getDataState(), iAgent, istatus, recv_state, reward);
  assert(iAgent>=0 && iAgent<static_cast<int>(agents.size()));

  sNew.set(recv_state);
  //agent's s is stored in sOld
  agents[iAgent]->Status = istatus;
  agents[iAgent]->swapStates();
  agents[iAgent]->setState(sNew);
  agents[iAgent]->getOldState(sOld);
  agents[iAgent]->getAction(aOld);
  agents[iAgent]->r = reward;
}

void Client::prepareAction(const int iAgent)
{
  if(iAgent<0) die("Error in iAgent number in Client::prepareAction\n");
  assert(iAgent >= 0 && iAgent < static_cast<int>(agents.size()));
  agents[iAgent]->act(aNew);
  double* const buf = comm->getDataAction();
  for (Uint i=0; i<aI.dim; i++) buf[i] = aNew.vals[i];
}
*/
