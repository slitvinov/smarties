//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Scheduler.h"
#include <fstream>
//#include <algorithm>
//#include <chrono>

namespace smarties
{

MasterSockets::MasterSockets(Settings& settings, DistributionInfo& distribinfo)
Master<MasterSockets>(settings, distribinfo) { }
MasterMPI::MasterMPI(Settings& settings, DistributionInfo& distribinfo)
Master<MasterMPI>(settings, distribinfo) { }

template<typename CommType>
void Master<CommType>::(Settings& settings, DistributionInfo& distribinfo) :
Worker(settings, distribinfo) { }

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

template<typename CommType>
void Master<CommType>::processCallers()
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

template<typename CommType>
void Master<CommType>::waitForStateActionCallers(const std::vector<Uint> givenWorkers)
{
  const size_t nClients = givenWorkers.size();
  std::vector<CommType::Request_t> reqs(nClients);
  // worker's rank is its index (givenWorkers[i]) plus 1 (master)
  for(size_t i=0; i<nClients; ++i) {
    const Uint callerID = givenWorkers[i]+1;
    const COMM_buffer& B = getCommBuffer( callerID );
    interface()->Irecv(B.dataStateBuf, B.sizeStateMsg, callerID, 0, reqs[i]);
  }

  for(size_t i=0; ; ++i) // infinite loop : communicate until break command
  {
    const Uint j = i % nClients, callID = givenWorkers[j], callRank = callID+1;
    // communication handle is rank_of_worker := workerID + 1 (master is 0)
    const int completed = interface()->TestComm(reqs[j]);
    //Learners lock workers if they have enough data to advance step
    while (bTrain && completed && learnersBlockingDataAcquisition()) {
      usleep(1); // this is to avoid burning cpus when waiting learners
      if(bExit.load()>0) break;
    }

    if(completed) {
      answerStateAction(callID);
      const COMM_buffer& B = getCommBuffer(callRank);
      interface()->Send(B.dataActionBuf, B.sizeActionMsg, callRank, 0);
      interface()->Irecv(B.dataStateBuf, B.sizeStateMsg,  callRank, 0, reqs[j]);
      if(bExit.load()>0) break;
    } else {
      usleep(1); // this is to avoid burning cpus when waiting environments
    }
  }

  for(const auto& A : agents) A->learnStatus = KILL;
  for(size_t i=0; i<nClients; ++i) { // send KILL messages
    interface()->WaitComm(reqs[i]);
    const COMM_buffer& B = getCommBuffer(givenWorkers[i]+1);
    interface()->Send(B.dataActionBuf, B.sizeActionMsg, givenWorkers[i]+1, 0);
  }
}

}
