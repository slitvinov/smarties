//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Worker.h"
#include "Learners/AllLearners.h"
#include "Utils/SocketsLib.h"
#include "Utils/SstreamUtilities.h"
#include <fstream>

namespace smarties
{

Worker::Worker(Settings&S, DistributionInfo&D) : settings(S), distrib(D),
  dataTasks( [&]() { return learnersBlockingDataAcquisition(); } ),
  algoTasks( [&]() { return learnersBlockingDataAcquisition(); } ),
  COMM( std::make_unique<Launcher>(this, D, S.bTrain) ),
  ENV( COMM->ENV ), agents( ENV.agents )
{
  /*
  if (0) { //}(distrib.bIsMaster) {
    // are we communicating with environments through sockets or mpi?
    //if(COMM->SOCK.clients.size()>0 == MPICommSize(master_workers_comm)>1);
    //  die("impossible: environments through mpi XOR sockets");
    if(distrib.nForkedProcesses2spawn > 0)
      assert(COMM->SOCK.clients.size() == (size_t) nCallingEnvs);
    else
      assert(MPICommSize(master_workers_comm) == (size_t) nCallingEnvs+1);
    assert(COMM->BUFF.size() == (size_t) nCallingEnvs);
  }
  */
}

void Worker::run()
{
  if(distrib.runInternalApp) // then worker lives inside the application
  {
    COMM->runApplication(distrib.environment_app_comm,
                         distrib.nWorker_processes,
                         distrib.thisWorkerGroupID);
  }
  else
  {
    COMM->forkApplication(distrib.nThreads, distrib.nOwnedEnvironments);
    synchronizeEnvironments();
    loopSocketsToMaster();
  }
}

void Worker::runTraining()
{
  const int learn_rank = MPICommRank(learners_train_comm);
  //////////////////////////////////////////////////////////////////////////////
  ////// FIRST SETUP SIMPLE FUNCTIONS TO DETECT START AND END OF TRAINING //////
  //////////////////////////////////////////////////////////////////////////////
  long minNdataB4Train = learners[0]->nObsB4StartTraining;
  int firstLearnerStart = 0, isStarted = 0, percentageReady = -5;
  for(Uint i=1; i<learners.size(); ++i)
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
    const Real factor = learners.size()==1? 1.0/ENV.nAgentsPerEnvironment : 1;
    const long dataCounter = bTrain? L->nLocTimeStepsTrain() : L->nSeqsEval();
    return dataCounter * factor >= settings.totNumSteps;
  };


  //////////////////////////////////////////////////////////////////////////////
  /////////////////////////// START DATA COLLECTION ////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  std::atomic<Uint> bDataCoordRunning {1};
  std::thread dataCoordProcess;

  #pragma omp parallel
  if( omp_get_thread_num() == std::min(omp_get_num_threads()-1, 2) )
    dataCoordProcess = std::thread( [&] () {
      while(1) {
        dataTasks.run();
        if (bDataCoordRunning == 0) break;
        usleep(1); // wait for workers to send data without burning a cpu
      }
    } );

  //////////////////////////////////////////////////////////////////////////////
  /////////////////////////////// TRAINING LOOP ////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  while(1) {
    isTrainingStarted();

    algoTasks.run();

    bool over = true;
    for(const auto& L : learners) over = over && isTrainingOver(L.get());
    if (over) break;
  }

  // kill data gathering process
  bDataCoordRunning = 0;
  dataCoordProcess.join();
}

void Worker::answerStateAction(const Uint bufferID) const
{
  assert( (Uint) bufferID < COMM->BUFF.size());
  const COMM_buffer& buffer = getCommBuffer(bufferID+1);
  const unsigned localAgentID = Agent::getMessageAgentID(buffer.dataStateBuf);
  // compute agent's ID within worker from the agentid within environment:
  const int agentID = bufferID * ENV.nAgentsPerEnvironment + localAgentID;
  //read from worker's buffer:
  assert( (Uint) agentID < agents.size() );
  assert( (Uint) agents[agentID]->workerID == bufferID );
  assert( (Uint) agents[agentID]->localID == localAgentID );
  Agent& agent = * agents[agentID].get();
  // unpack state onto agent
  agent.unpackStateMsg(buffer.dataStateBuf);
  if(agent.agentStatus == FAIL) die("app crashed. TODO: handle");
  // get learning algorithm:
  Learner& algo = * learners[getLearnerID(localAgentID)].get();
  //pick next action and ...do a bunch of other stuff with the data:
  algo.select(agent);

  //static constexpr auto vec2str = Utilities::vec2string<double>;
  //const int agentStatus = status2int(agent.agentStatus);
  //_warn("Agent %d %d:[%s]>[%s] r:%f a:[%s]", agentID, agentStatus,
  //      vec2str(agent.sOld,-1).c_str(), vec2str(agent.state,-1).c_str(),
  //      agent.reward, vec2str(agent.action,-1).c_str());

  // Some logging and passing around of step id:
  const Real factor = learners.size()==1? 1.0/ENV.nAgentsPerEnvironment : 1;
  const Uint nSteps = algo.nLocTimeStepsTrain();
  agent.learnerStepID = factor * nSteps;
  agent.packActionMsg(buffer.dataActionBuf);
  if(agent.agentStatus >= TERM) // localAgentID, bufferID
    dumpCumulativeReward(agent, algo.nGradSteps(), nSteps);
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
  if (learnAlgoIter == 0 && bTrain) return;
  const int wrank = MPICommSize(MPI_COMM_WORLD);
  char path[256];
  sprintf(path, "agent_%02u_rank%02d_cumulative_rewards.dat", agent.localID, wrank);

  std::lock_guard<std::mutex> lock(dump_mutex);
  FILE * pFile = fopen (path, "a");
  fprintf (pFile, "%lu %lu %u %u %f\n", learnAlgoIter, totalAgentTstep,
    agent.workerID, agent.timeStepInEpisode, agent.cumulativeRewards);
  fflush (pFile);
  fclose (pFile);
}

void Worker::synchronizeEnvironments()
{
  // here cannot use the recurring template because behavior changes slightly:
  const std::function<void(void*, size_t)> recvBuffer = [&](void* buffer, size_t size)
  {
    bool received = false;
    if( COMM->SOCK.clients.size() > 0 ) { // master with apps connected through sockets (on the same compute node)
      SOCKET_Brecv(buffer, size, COMM->SOCK.clients[0]);
      received = true;
      for(size_t i=1; i < COMM->SOCK.clients.size(); ++i) {
        void * const testbuf = malloc(size);
        SOCKET_Brecv(testbuf, size, COMM->SOCK.clients[i]);
        const int err = memcmp(testbuf, buffer, size); free(testbuf);
        if(err) die(" error: comm mismatch");
      }
    }

    if(master_workers_comm not_eq MPI_COMM_NULL)
    if( MPICommSize(master_workers_comm) >  1 &&
        MPICommRank(master_workers_comm) == 0 ) {
      if(received) die("Sockets and MPI workers: should be impossible");
      MPI_Recv(buffer, size, MPI_BYTE, 1, 368637, master_workers_comm, MPI_STATUS_IGNORE);
      received = true;
      // size of comm is number of workers plus master:
      for(Uint i=2; i < MPICommSize(master_workers_comm); ++i) {
        void * const testbuf = malloc(size);
        MPI_Recv(testbuf, size, MPI_BYTE, i, 368637, master_workers_comm, MPI_STATUS_IGNORE);
        const int err = memcmp(testbuf, buffer, size); free(testbuf);
        if(err) die(" error: mismatch");
      }
    }

    if(master_workers_comm not_eq MPI_COMM_NULL)
    if( MPICommSize(master_workers_comm) >  1 &&
        MPICommRank(master_workers_comm) >  0 ) {
      MPI_Send(buffer, size, MPI_BYTE, 0, 368637, master_workers_comm);
    }

    if(workerless_masters_comm == MPI_COMM_NULL) return;
    assert( MPICommSize(workerless_masters_comm) >  1 );

    if( MPICommRank(workerless_masters_comm) == 0 ) {
      if(not received) die("rank 0 of workerless masters comm has no worker");
      for(Uint i=1; i < MPICommSize(workerless_masters_comm); ++i)
        MPI_Send(buffer, size, MPI_BYTE, i, 368637, workerless_masters_comm);
    }

    if( MPICommRank(workerless_masters_comm) >  0 ) {
      if(received) die("rank >0 of workerless masters comm owns workers");
      MPI_Recv(buffer, size, MPI_BYTE, 0, 368637, workerless_masters_comm, MPI_STATUS_IGNORE);
    }
  };

  ENV.synchronizeEnvironments(recvBuffer, distrib.nOwnedEnvironments);

  for(Uint i=0; i<ENV.nAgents; ++i) COMM->initOneCommunicationBuffer();

  // now i know nAgents, might need more generators:
  distrib.finalizePRNG(ENV.nAgents);
  distrib.nAgents = ENV.nAgents;

  // return if this process should not host the learning algorithms
  if(not distrib.bIsMaster and not distrib.learnersOnWorkers) return;

  const Uint nAlgorithms =
    ENV.bAgentsHaveSeparateMDPdescriptors? ENV.nAgentsPerEnvironment : 1;
  distrib.nOwnedAgentsPerAlgo =
    distrib.nOwnedEnvironments * ENV.nAgentsPerEnvironment / nAlgorithms;
  learners.reserve(nAlgorithms);
  for(Uint i = 0; i<nAlgorithms; ++i)
  {
    char lName[256]; sprintf(lName, "agent_%02lu", i);
    if(distrib.world_rank == 0) printf("Learner: %s\n", lName);
    learners.emplace_back( createLearner(ENV.getDescriptor(i), settings, distrib) );
    assert(learners.size() == i+1);
    learners[i]->setLearnerName(std::string(lName)+"_", i);
    learners[i]->restart();
    learners[i]->setupTasks(algoTasks);
    learners[i]->setupDataCollectionTasks(dataTasks);
  }
}

void Worker::loopSocketsToMaster()
{
  bool bTerminate = false;
  const size_t nClients = COMM->SOCK.clients.size();
  std::vector<SOCKET_REQ> reqs = std::vector<SOCKET_REQ>(nClients);
  // worker's communication functions behave following mpi indexing
  // sockets's rank (bufferID) is its index plus 1 (master)
  for(size_t i=0; i<nClients; ++i) {
    const auto& B = getCommBuffer(i+1); const int SID = getSocketID(i+1);
    SOCKET_Irecv(B.dataStateBuf, B.sizeStateMsg, SID, reqs[i]);
  }

  for(size_t i=0; ; ++i) // infinite loop : communicate until break command
  {
    const int workID = i % nClients, SID = getSocketID(workID+1);
    const auto& B = getCommBuffer(workID+1);

    SOCKET_Test(reqs[workID].completed, reqs[workID]);

    if(reqs[workID].completed) {
      stepWorkerToMaster(workID);
      learnerStatus& S = Agent::messageLearnerStatus((char*) B.dataActionBuf);
      // check if abort was called. don't tell the app yet, cleaner loop later
      if(S == KILL) { bTerminate = true; S = WORK; }
      SOCKET_Bsend(B.dataActionBuf, B.sizeActionMsg, SID);
    }
    else usleep(1); // wait for app to send a state without burning a cpu

    if(bTerminate) break;
  }

  for(size_t i=0; i<nClients; ++i) {
    SOCKET_Wait(reqs[i]);
    const auto& B = getCommBuffer(i+1);
    Agent::messageLearnerStatus((char*) B.dataActionBuf) = KILL;
    SOCKET_Bsend(B.dataActionBuf, B.sizeActionMsg, getSocketID(i+1));
  }
}

void Worker::stepWorkerToMaster(const Uint bufferID) const
{
  assert(master_workers_comm not_eq MPI_COMM_NULL);
  assert(MPICommRank(master_workers_comm) > 0 || learners.size()>0);

  if(learners.size()) return answerStateAction(bufferID);

  const COMM_buffer& BUF = getCommBuffer(bufferID+1);
  const auto& appCom = COMM->workers_application_comm;
  const int appRank = MPICommRank(appCom), appSize = MPICommSize(appCom);
  if(appSize) assert( COMM->SOCK.clients.size() == 0 );

  if (appRank<=0 || COMM->bEnvDistributedAgents)
  {
    // MPI MSG to master of a single state:
    MPI_Request send_request, recv_request;
    MPI_Isend(BUF.dataStateBuf, BUF.sizeStateMsg, MPI_BYTE,
        0, 78283, master_workers_comm, &send_request);
    MPI_Request_free(&send_request);
    // MPI MSG from master of a single action:
    MPI_Irecv(BUF.dataActionBuf, BUF.sizeActionMsg, MPI_BYTE,
        0, 22846, master_workers_comm, &recv_request);
    while (1) {
      int completed = 0;
      MPI_Test(&recv_request, &completed, MPI_STATUS_IGNORE);
      if (completed) break;
      usleep(1); // wait action from master without burning cpu resources
    }

    if (not COMM->bEnvDistributedAgents && appSize>1) {
      //Then this is rank 0 of an environment with centralized agents.
      //Broadcast same action to members of the gang:
      MPI_Bcast(BUF.dataActionBuf, BUF.sizeActionMsg, MPI_BYTE, 0, appCom);
    }
  }
  else // rank_inside_app>0 && not bEnvDistributedAgents
  {
    //Else this function was called by rank>0 of an app with centralized agents.
    //Therefore, recv the action obtained from master:
    MPI_Bcast(BUF.dataActionBuf, BUF.sizeActionMsg, MPI_BYTE, 0, appCom);
  }
  //learner_step_id = (unsigned) BUF.dataActionBuf[0];
}

int Worker::getSocketID(const Uint worker) const
{
  assert( worker <= COMM->SOCK.clients.size() );
  return worker>0? COMM->SOCK.clients[worker-1] : COMM->SOCK.server;
}

const COMM_buffer& Worker::getCommBuffer(const Uint worker) const
{
  assert( worker>0 && worker <= COMM->BUFF.size() );
  return * COMM->BUFF[worker-1].get();
}

}
