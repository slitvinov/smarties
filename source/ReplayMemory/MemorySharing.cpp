//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemorySharing.h"
#include "Utils/FunctionUtilities.h"

namespace smarties
{

MemorySharing::MemorySharing(MemoryBuffer*const RM) : replay(RM)
{
  completed.reserve(distrib.nAgents);

  if(distrib.workerless_masters_comm == MPI_COMM_NULL &&
     distrib.learnersOnWorkers == false) return;
  if(distrib.workerless_masters_comm != MPI_COMM_NULL &&
     distrib.learnersOnWorkers) die("TODO: workerlessmasters & workerlearner");

  if(distrib.workerless_masters_comm != MPI_COMM_NULL) {
    sharingComm = MPICommDup( distrib.workerless_masters_comm);
    sharingSize = MPICommSize(workerComm);
    sharingRank = MPICommRank(workerComm);
  }

  if(distrib.learnersOnWorkers)
  {
    if(distrib.master_workers_comm == MPI_COMM_NULL) {
      warn("learning algorithm entirely hosted on workers"); return;
    }
    // rank>0 (worker) will always send to rank==0 (master)
    workerComm = MPICommDup( distrib.master_workers_comm);
    workerSize = MPICommSize(workerComm);
    workerRank = MPICommRank(workerComm);
    if(workerSize < 2) {
      warn("detected no workers in the wrong spot..."); return;
    }
    if(sharingRank == 0) {
      if(not distrib.bIsMaster) die("impossible");
      workerRecvSizeReq = std::vector<MPI_Request>(workerSize);
      workerRecvSeqSize = std::vector<unsigned long>(workerSize);
      //workerRecvSeq = std::vector<Fvec>(workerSize);
    } else {
      if(distrib.bIsMaster) die("impossible");
      return; // do not create thread
    }
  } else {
    workerSize = 0; workerRank = 0; workerComm = MPI_COMM_NULL;
  }

  if(distrib.workerless_masters_comm != MPI_COMM_NULL)
  {
    sharingTurn = sharingRank; // says that first full episode stays on rank
    shareSendSizeReq = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareRecvSizeReq = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareSendSeqReq  = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareSendSeqSize = std::vector<unsigned long>(sharingSize);
    shareRecvSeqSize = std::vector<unsigned long>(sharingSize);
    shareSendSeq = std::vector<Fvec>(sharingSize);
    //shareRecvSeq = std::vector<Fvec>(sharingSize);
  } else {
    sharingSize = 0; sharingRank = 0; sharingComm = MPI_COMM_NULL;
  }

  //if(distrib.bIsMaster) { } else { }
  bFetcherRunning = 1;
  #pragma omp parallel
  {
    const int thrID = omp_get_thread_num();
    const int tgtCPU =  std::min(omp_get_num_threads()-1, 2);
    if( thrID==tgtCPU ) fetcher = std::thread( [ &, this ] () { run(); } );
  }
}

MemorySharing::~MemorySharing()
{
  if(bFetcherRunning) {
    bFetcherRunning = 0;
    fetcher.join();
  }

  for(auto & S : completed) Utilities::dispose_object(S);
}

void MemorySharing::run()
{
  for(Uint i=1; i<workerSize; i++)
    IrecvSize(workerRecvSeqSize[i], i, workerComm, workerRecvSizeReq[i]);

  for(Uint i=0; i<sharingSize; i++)
    if(i not_eq sharingRank)
      IrecvSize(shareRecvSeqSize[i], i, sharingComm, shareRecvSizeReq[i]);

  while(true)
  {
    {
      std::lock_guard<std::mutex> lock(complete_mutex);
      while ( completed.size() )
      {
        if (sharingTurn == sharingRank)
          replay->pushBackSequence(completed.back());
        else
        {
          const Uint I = sharingTurn;
          Sequence* const EP = completed.back();
          if(shareSendSizeReq[I] not_eq MPI_REQUEST_NULL)
            MPI(Wait, & shareSendSizeReq[I], MPI_STATUS_IGNORE);
          if( shareSendSeqReq[I] not_eq MPI_REQUEST_NULL)
            MPI(Wait, &  shareSendSeqReq[I], MPI_STATUS_IGNORE);

          shareSendSeq[I] = EP->packSequence(sI.dimObs(),aI.dim(),aI.dimPol());
          Utilities::dispose_object(completed.back());
          shareSendSeqSize[I] = shareSendSeq[I].size();

          IsendSize(shareSendSeqSize[I], I, sharingComm, shareSendSizeReq[I]);
          IsendSeq(shareSendSeq[I], I, sharingComm, shareSendSeqReq[I]);
        }
        completed.pop_back();
        // who's turn is next to receive an episode?
        sharingTurn = (sharingTurn+1) % sharingSize;
      }
    }

    for(Uint i=1; i<workerSize; i++)
      if (isComplete(workerRecvSizeReq[i]))
      {
        Fvec EP(workerRecvSeqSize[i], (Fval)0);
        RecvSeq(EP, i, workerComm);
        // prepare the next one:
        IrecvSize(workerRecvSeqSize[i], i, workerComm, workerRecvSizeReq[i]);
        if (sharingTurn == sharingRank) {
          Sequence * const tmp = new Sequence();
          tmp->unpackSequence(EP, sI.dimObs(), aI.dim(), aI.dimPol());
          replay->pushBackSequence(tmp);
        } else {
          const Uint I = sharingTurn;
          if(shareSendSizeReq[I] not_eq MPI_REQUEST_NULL)
            MPI(Wait, & shareSendSizeReq[I], MPI_STATUS_IGNORE);
          if( shareSendSeqReq[I] not_eq MPI_REQUEST_NULL)
            MPI(Wait, &  shareSendSeqReq[I], MPI_STATUS_IGNORE);

          shareSendSeq[I] = EP;
          shareSendSeqSize[I] = shareSendSeq[I].size();

          IsendSize(shareSendSeqSize[I], I, sharingComm, shareSendSizeReq[I]);
          IsendSeq(shareSendSeq[I], I, sharingComm, shareSendSeqReq[I]);
        }
        sharingTurn = (sharingTurn+1) % sharingSize;
      }

    for(Uint i=0; i<sharingSize; i++)
      if (isComplete(shareRecvSizeReq[i]))
      {
        Fvec EP(shareRecvSeqSize[i], (Fval)0);
        RecvSeq(EP, i, sharingComm);
        // prepare the next one:
        IrecvSize(shareRecvSeqSize[i], i, sharingComm, shareRecvSizeReq[i]);
        Sequence * const tmp = new Sequence();
        tmp->unpackSequence(EP, sI.dimObs(), aI.dim(), aI.dimPol());
        replay->pushBackSequence(tmp);
      }

    usleep(1);
    if( bFetcherRunning.load() == 0 ) break;
  }
}

void MemorySharing::addComplete(Sequence* const EP)
{
  if(sharingComm not_eq MPI_COMM_NULL)
  {
    std::lock_guard<std::mutex> lock(complete_mutex);
    completed.push_back(EP);
  }
  else if(workerComm not_eq MPI_COMM_NULL)
  {
    // if we created data structures for worker to send eps to master
    // this better be a worker!
    assert(workerRank>0 && workerSize>1 && not distrib.bIsMaster);
    Fvec sendSq = EP->packSequence(sI.dimObs(), aI.dim(), aI.dimPol());
    unsigned long sendSz = sendSq.size();
    MPI(Send, &sendSz, 1, MPI_UNSIGNED_LONG, 0, 99, workerComm);
    MPI(Send, sendSq.data(), sendSz, MPI_Fval, 0, 98, workerComm);
  }
  else // data stays here
  {
    replay->pushBackSequence(EP);
  }
}

}
