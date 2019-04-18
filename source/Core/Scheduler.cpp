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

Worker::Worker(Settings& settings, DistributionInfo& distrib)
: comm(_c), env(_e)
{
}

void Worker::runTraining()
{
  //////////////////////////////////////////////////////////////////////////////
  ////// FIRST SETUP SIMPLE FUNCTIONS TO DETECT START AND END OF TRAINING //////
  //////////////////////////////////////////////////////////////////////////////
  Uint minNdataB4Train = learners[0]->nObsB4StartTraining;
  int firstLearnerStart = 0, isStarted = 0, percentageReady = -5;
  for(size_t i=1; i<learners.size(); i++)
    if(learners[i]->nObsB4StartTraining < minNdataB4Train) {
      minNdataB4Train = learners[i]->nObsB4StartTraining;
      firstLearnerStart = i;
    }

  const auto isTrainingStarted = [&]() {
    if(not isStarted && learn_rank==0) {
      const auto nCollected = learners[firstLearnerStart]->locDataSetSize();
      const int currPerc = nCollected * 100./(Real) minNdataB4Train;
      if(nCollected >= minNdataB4Train) isStarted = true;
      else if(currPerc > percentageReady+5) {
       percentageReady = currPerc;
       printf("\rCollected %d%% of data required to begin training. ",currPerc);
       fflush(0);
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

  //////////////////////////////////////////////////////////////////////////////
  /////////////////////////////// TRAINING LOOP ////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  while(1) {
    isTrainingStarted();

    tasks.run();

    bool over = true;
    for(const auto& L : learners) over = over && isTrainingOver(L);
    if (over) break;
  }
}

void Master::processCallers()
{
  // are we communicating with environments through sockets or mpi?
  assert(COMM.SOCK.clients.size()>0 not_eq MPICommSize(master_workers_comm)>1);
    die("impossible: environments through mpi XOR sockets");
  if(mpiProcessing)
    assert(MPICommSize(master_workers_comm) == (size_t) nCallingEnvs+1);
  if(socketsProcessing)
    assert(COMM.SOCK.clients.size() == (size_t) nCallingEnvs);
  assert(COMM.BUFF.size() == (size_t) nCallingEnvs);

  #pragma omp parallel num_threads(nThreads)
  {
    std::vector<Uint> shareWorkers;
    const Uint thrN = omp_get_num_threads();
    const Uint thrID = thrN-1 - omp_get_thread_num(); // thrN-1, thrN-2, ..., 0
    const Uint workerShare = std::ceil(nCallingEnvs / (double) thrN);
    const Uint workerBeg = thrID * workerShare;
    const Uint workerEnd = std::min(nCallingEnvs, (thrID+1)*workerShare);
    for(Uint i=workerBeg; i<workerEnd; i++) shareWorkers.push_back(i);

    #pragma omp critical
    if (shareWorkers.size())
      worker_replies.push_back (
        std::thread( [&, shareWorkers] () {
          waitForStateActionCallers(shareWorkers); } ) );
  }
}

void Master::waitForStateActionCallers(const std::vector<Uint> givenWorkers)
{
  const size_t nClients = givenWorkers.size();
  // worker's rank is its index (givenWorkers[i]) plus 1 (master)
  for(size_t i=0; i<nClients; ++i) interface()->IrecvState(givenWorkers[i]+1);

  for(size_t i=0; ; ++i) // infinite loop : communicate until break command
  {
    const Uint workerID = givenWorkers[ i % nClients ];
    // communication handle is rank_of_worker := workerID + 1 (master is 0)
    const int completed = interface()->TestComm(workerID+1);
    //Learners lock workers if they have enough data to advance step
    while (bTrain && completed && learnersBlockingDataAcquisition()) {
      usleep(1); // this is to avoid burning cpus when waiting learners
      if(bExit.load()>0) break;
    }

    if(completed) {
      answerStateAction(workerID);
      interface()->SendAction(workerID+1);
      interface()->IrecvState(workerID+1);
      if(bExit.load()>0) break;
    } else {
      usleep(1); // this is to avoid burning cpus when waiting environments
    }
  }

  for(const auto& A : agents) A->learnStatus = KILL;
  for(size_t i=0; i<nClients; ++i) {
    interface()->WaitComm(givenWorkers[i]+1);
    interface()->SendAction(givenWorkers[i]+1); // send KILL message
  }
}

void Worker::answerStateAction(const int bufferID)
{
  assert(bufferID < COMM.BUFF.size());
  const COMM_buffer& buffer = * COMM.BUFF[bufferID].get();
  const unsigned localAgentID = Agent::getMessageAgentID(buffer.dataStateBuf);
  // compute agent's ID within worker from the agentid within environment:
  const int agentID = bufferID * ENV.nAgentsPerEnvironment + localAgentID;
  //read from worker's buffer:
  assert(agentID < agents.size());
  Agent& agent = * agents[agentID].get();
  agent.unpackStateMsg(buffer.dataStateBuf);
  if(agent.agentStatus == FAIL) die("app crashed. TODO: handle");
  // get learning algorithm:
  Learner& algo = * learners[getLearnerID(localAgentID)].get();
  //pick next action and ...do a bunch of other stuff with the data:
  algo.select(agent);
  //debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]",
  //  agent, agents[agent]->Status, agents[agent]->sOld._print().c_str(),
  //  agents[agent]->s._print().c_str(), agents[agent]->r,
  //  agents[agent]->a._print().c_str());

  // Some logging and passing around of step id:
  const Real factor = learners.size()==1? 1.0/ENV.nAgentsPerEnvironment : 1;
  const Uint nSteps = algo.nLocTimeStepsTrain();
  agent.learnerStepID = factor * nSteps;
  if(agent.agentStatus >= TERM)
    dumpCumulativeReward(localAgentID, bufferID, algo.nGradSteps(), nSteps);
  //debugS("Sent action to worker %d: [%s]", worker, print(actVec).c_str() );
}

Uint Worker::getLearnerID(const Uint agentIDlocal) const
{
  // some asserts:
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

bool Worker::learnersBlockingDataAcquisition() const
{
  //When would a learning algo stop acquiring more data?
  //Off Policy algos:
  // - User specifies a ratio of observed trajectories to gradient steps.
  //    Comm is restarted or paused to maintain this ratio consant.
  //On Policy algos:
  // - if collected enough trajectories for current batch, then comm is paused
  //    untill gradient is applied (or nepocs are done), then comm restarts
  //    to obtain fresh on policy samples
  // However, no learner can stop others from getting data (vector of algos)
  bool lock = true;
  for (const auto& L : learners) lock = lock && L->blockDataAcquisition();
  return lock;
}

void Worker::dumpCumulativeReward(const Agent& agent,
  const Uint learnAlgoIter, const Uint totalAgentTstep) const
{
  if (giter == 0 && bTrain) return;
  const int wrank = MPICommSize(MPI_COMM_WORLD);
  char path[256];
  sprintf(path, "agent_%02d_rank%02d_cumulative_rewards.dat", agent.localID, wrank);

  std::lock_guard<std::mutex> lock(dump_mutex);
  FILE * pFile = fopen (path, "a");
  fprintf (pFile, "%u %u %d %d %f\n", learnAlgoIter, totalAgentTstep,
    agent.workerID, agent.transitionID, agent.cumulative_rewards);
  fflush (pFile);
  fclose (pFile);
}

void Worker::synchronizeEnvironments()
{
  // here cannot use the recurring template because behavior changes slightly:
  const std::function<void(void*, size_t)> recvBuffer = [&]()
  {
    bool received = false;
    if( COMM.SOCK.clients.size() > 0 ) { // master with apps connected through sockets (on the same compute node)
      SOCKET_Recv(buffer, size, COMM.SOCK.clients[0]);
      received = true;
      for(size_t i=1; i < COMM.SOCK.clients.size(); ++i) {
        void * const testbuf = malloc(size);
        SOCKET_Recv(testbuf, size, COMM.SOCK.clients[i]);
        const int err = memcmp(testbuf, buffer, size); free(buffer);
        if(err) die(" error: comm mismatch");
      }
    }

    if( MPICommSize(master_workers_comm) >  1 &&
        MPICommRank(master_workers_comm) == 0 ) {
      if(received) die("Sockets and MPI workers: should be impossible");
      MPI_Recv(buffer, size, MPI_BYTE, 1, 368637, master_workers_comm, MPI_STATUS_IGNORE);
      received = true;
      // size of comm is number of workers plus master:
      for(Uint i=2; i < MPICommSize(master_workers_comm); ++i) {
        void * const testbuf = malloc(size);
        MPI_Recv(testbuf, size, MPI_BYTE, i, 368637, master_workers_comm, MPI_STATUS_IGNORE);
        const int err = memcmp(testbuf, buffer, size); free(buffer);
        if(err) die(" error: mismatch");
      }
    }

    if( MPICommSize(master_workers_comm) >  1 &&
        MPICommRank(master_workers_comm) >  0 ) {
      MPI_Send(buffer, size, MPI_BYTE, 0, 368637, master_workers_comm);
    }

    if( MPICommSize(workerless_masters_comm) >  1 &&
        MPICommRank(workerless_masters_comm) == 0 ) {
      if(not received) die("rank 0 of workerless masters comm has no worker");
      for(Uint i=1; i < MPICommSize(workerless_masters_comm); ++i)
        MPI_Send(buffer, size, MPI_BYTE, i, 368637, workerless_masters_comm);
    }

    if( MPICommSize(workerless_masters_comm) >  1 &&
        MPICommRank(workerless_masters_comm) >  0 ) {
      if(received) die("rank >0 of workerless masters comm owns workers");
      MPI_Recv(buffer, size, MPI_BYTE, 0, 368637, master_workers_comm, MPI_STATUS_IGNORE);
    }
  };

  synchronizeEnvironments(recvBuffer, distrib.nOwnedEnvironments);

  for(Uint i=0; i<distrib.nOwnedEnvironments; ++i)
    COMM.initOneCommunicationBuffer();

  // now i know nAgents, might need more generators:
  distrib.finalizePRNG(ENV.nAgents);

  // return if this process should not host the learning algorithms
  if(not distrib.bIsMaster and not distrib.learnersOnWorkers) return;

  const Uint nLearners = ENV.bAgentsHaveSeparateMDPdescriptors? 1 : ENV.nAgentsPerEnvironment;
  learners.reserve(nLearners);
  for(Uint i = 0; i<nLearners; i++)
  {
    std::stringstream ss; ss<<"agent_"<<std::setw(2)<<std::setfill('0')<<i;
    if(distrib.world_rank == 0) printf("Learner: %s\n", ss.str().c_str());
    learners[i] = createLearner(ENV, settings);
    learners[i]->setLearnerName(ss.str() +"_", i);
    learners[i]->restart();
  }
}

void Worker::loopSocketToMaster() const
{
  bool bTerminate = false;
  const size_t nClients = COMM.SOCK.clients.size();
  std::vector<SOCKET_REQ> reqs = std::vector<SOCKET_REQ>(nClients);
  // worker's communication functions behave following mpi indexing
  // sockets's rank (bufferID) is its index plus 1 (master)
  for(size_t i=0; i<nClients; ++i) {
    const auto& B = getCommBuffer(i+1); const int SID = getSocketID(i+1);
    SOCKET_Irecv(B.dataStateMsg, B.sizeStateMsg, SID, reqs[i]);
  }

  for(size_t i=0; ; ++i) // infinite loop : communicate until break command
  {
    const int workID = i % nClients, SID = getSocketID(workID+1);
    const auto& B = getCommBuffer(workID+1);

    SOCKET_Test(reqs[workID].completed, reqs[workID]);

    if(reqs[workID].completed) {
      stepWorkerToMaster(workID);
      learnerStatus& lstatus = messageLearnerStatus(B.dataActionMsg);
      // check if abort was called. don't tell the app yet, cleaner loop later
      if(lstatus == KILL) { bTerminate = true; lstatus = WORK; }
      SOCKET_Bsend(B.dataActionMsg, B.sizeActionMsg, SID);
    }
    else usleep(1); // wait for app to send a state without burning a cpu

    if(bTerminate) break;
  }

  for(const auto& A : agents) A->learnStatus = KILL;
  for(size_t i=0; i<nClients; ++i) {
    SOCKET_Wait(reqs[i]);
    SOCKET_Bsend(B.dataActionMsg, B.sizeActionMsg, getSocketID(i+1));
  }
}

void Worker::stepWorkerToMaster(const Uint bufferID) const
{
  assert(master_workers_comm not_eq MPI_COMM_NULL);
  assert(MPICommRank(master_workers_comm) > 0 || learners.size()>0);
  const COMM_buffer& BUF = getCommBuffer(bufferID+1);
  const auto& appCom = COMM.workers_application_comm;
  const int appRank = MPICommRank(appCom), appSize = MPICommSize(appCom);
  if(appSize) assert( COMM.SOCK.clients.size() == 0 );

  if(learners.size()) return answerStateAction();

  if (appRank<=0 || COMM.bEnvDistributedAgents)
  {
    // MPI MSG to master of a single state:
    MPI_Request send_request, recv_request;
    MPI_Isend(BUF.dataStateMsg, BUF.sizeStateMsg, MPI_BYTE,
        0, 22846, master_workers_comm, &send_request);
    MPI_Request_free(&send_request);
    // MPI MSG from master of a single action:
    MPI_Irecv(BUF.dataActionMsg, BUF.sizeActionMsg, MPI_BYTE,
        0, 22846, master_workers_comm, &recv_request);
    while (1) {
      int completed = 0;
      MPI_Test(&recv_request, &completed, MPI_STATUS_IGNORE);
      if (completed) break;
      usleep(1); // wait action from master without burning cpu resources
    }

    if (not COMM.bEnvDistributedAgents && appSize>1) {
      //Then this is rank 0 of an environment with centralized agents.
      //Broadcast same action to members of the gang:
      MPI_Bcast(BUF.dataActionMsg, BUF.sizeActionMsg, MPI_BYTE, 0, appCom);
    }
  }
  else // rank_inside_app>0 && not bEnvDistributedAgents
  {
    //Else this function was called by rank>0 of an app with centralized agents.
    //Therefore, recv the action obtained from master:
    MPI_Bcast(BUF.dataActionMsg, BUF.sizeActionMsgg, MPI_BYTE, 0, appCom);
  }

  //learner_step_id = (unsigned) BUF.dataActionMsg[0];
}
