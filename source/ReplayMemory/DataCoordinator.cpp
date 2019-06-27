//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "DataCoordinator.h"
#include "Utils/FunctionUtilities.h"


namespace smarties
{

DataCoordinator::DataCoordinator(MemoryBuffer*const RM, ParameterBlob & P)
  : replay(RM), params(P)
{
  completed.reserve(distrib.nAgents);

  // if all masters have socketed workers, no need for the coordinator
  if(distrib.workerless_masters_comm == MPI_COMM_NULL &&
     distrib.learnersOnWorkers == false) return;

  if(distrib.workerless_masters_comm not_eq MPI_COMM_NULL) {
    sharingComm = MPICommDup( distrib.workerless_masters_comm);
    sharingSize = MPICommSize(sharingComm);
    sharingRank = MPICommRank(sharingComm);
  }

  if(distrib.learnersOnWorkers && distrib.nOwnedEnvironments) {
    //warn("Creating communicator to send episodes from workers to learners.");
    if(distrib.master_workers_comm == MPI_COMM_NULL) {
      warn("learning algorithm entirely hosted on workers"); return;
    }
    workerComm = MPICommDup(distrib.master_workers_comm);
    // rank>0 (worker) will always send to rank==0 (master)
    workerSize = MPICommSize(workerComm);
    workerRank = MPICommRank(workerComm);
    if(workerSize < 2) {
      warn("detected no workers in the wrong spot..."); return;
    }
    if(workerRank == 0) {
      if(not distrib.bIsMaster) die("impossible");
      workerRecvSizeReq = std::vector<MPI_Request>(workerSize);
      workerRecvSeqSize = std::vector<unsigned long>(workerSize);
      workerReqParamFlag= std::vector<unsigned long>(workerSize);
      workerReqParamReq = std::vector<MPI_Request>(workerSize);
      //workerRecvSeq = std::vector<Fvec>(workerSize);
    } else {
      if(distrib.bIsMaster) die("impossible");
    }
  } else {
    workerSize = 0; workerRank = 0; workerComm = MPI_COMM_NULL;
  }

  if(distrib.workerless_masters_comm not_eq MPI_COMM_NULL) {
    warn("Creating communicator for learners without workers to recv episodes from learners with workers.");
    sharingTurn = sharingRank; // says that first full episode stays on rank
    shareSendSizeReq = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareRecvSizeReq = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareSendSeqReq  = std::vector<MPI_Request>(sharingSize, MPI_REQUEST_NULL);
    shareSendSeqSize = std::vector<unsigned long>(sharingSize);
    shareRecvSeqSize = std::vector<unsigned long>(sharingSize);
    shareSendSeq = std::vector<Fvec>(sharingSize);
    //shareRecvSeq = std::vector<Fvec>(sharingSize);
  } else {
    sharingTurn = 0; sharingSize = 0; sharingRank = 0; sharingComm = MPI_COMM_NULL;
  }
}

DataCoordinator::~DataCoordinator()
{
  if(sharingComm not_eq MPI_COMM_NULL) MPI_Comm_free(&sharingComm);
  if(workerComm not_eq MPI_COMM_NULL) MPI_Comm_free(&workerComm);
  for(auto & S : completed) Utilities::dispose_object(S);
}

void DataCoordinator::setupTasks(TaskQueue& tasks)
{
  allTasksPtr = & tasks;
  if(workerRank > 0) return;

  //////////////////////////////////////////////////////////////////////
  // Waiting for workers to request parameters
  /////////////////////////////////////////////////////////////////////
  assert(workerReqParamFlag.size() == workerSize);
  assert(workerReqParamReq .size() == workerSize);
  //for(Uint i=1; i<workerSize; ++i)
  //  MPI(Irecv, & workerReqParamFlag[i], 1, MPI_UNSIGNED_LONG, i, 89,
  //             workerComm, & workerReqParamReq[i]);

  //////////////////////////////////////////////////////////////////////
  // Waiting for workers to send episodes
  /////////////////////////////////////////////////////////////////////
  for(Uint i=1; i<workerSize; ++i)
    IrecvSize(workerRecvSeqSize[i], i, workerComm, workerRecvSizeReq[i]);

  //////////////////////////////////////////////////////////////////////
  // Waiting for other masters to share episodes
  /////////////////////////////////////////////////////////////////////
  for(Uint i=0; i<sharingSize; ++i)
    if(i not_eq sharingRank)
      IrecvSize(shareRecvSeqSize[i], i, sharingComm, shareRecvSizeReq[i]);

  //if (workerSize > 0)
  //{
  //  auto stepSendParams = [&] () {
  //    answerWorkersParameterUpdates();
  //  };
  //  tasks.add(stepSendParams);
  //}
  if (sharingSize > 0 || workerSize > 0)
  {
    auto stepDistribEps = [&] () {
      distributePendingEpisodes();
    };
    tasks.add(stepDistribEps);
  }
  if (sharingSize > 0 || workerSize > 0)
  {
    auto stepReceiveEps = [&] () {
      mastersRecvEpisodes();
    };
    tasks.add(stepReceiveEps);
  }
}

void DataCoordinator::answerWorkersParameterUpdates()
{
  for(Uint i=1; i<workerSize; ++i)
    if (isComplete(workerReqParamReq[i])) {
      //_warn("send parameters to rank %lu", i);
      params.send(i, 0);
      MPI(Irecv, & workerReqParamFlag[i], 1, MPI_UNSIGNED_LONG, i, 89,
               workerComm, & workerReqParamReq[i]);

    }
}

void DataCoordinator::distributePendingEpisodes()
{
  std::lock_guard<std::mutex> lockQueue(complete_mutex);
  while ( completed.size() )
  {
    Sequence* const EP = completed.back();
    if (sharingTurn == sharingRank) replay->pushBackSequence(EP);
    else
    {
      const Uint I = sharingTurn;
      if(shareSendSizeReq[I] not_eq MPI_REQUEST_NULL)
        MPI(Wait, & shareSendSizeReq[I], MPI_STATUS_IGNORE);
      if( shareSendSeqReq[I] not_eq MPI_REQUEST_NULL)
        MPI(Wait, &  shareSendSeqReq[I], MPI_STATUS_IGNORE);

      shareSendSeq[I] = EP->packSequence(sI.dimObs(),aI.dim(),aI.dimPol());
      shareSendSeqSize[I] = shareSendSeq[I].size();
      Utilities::dispose_object(EP);

      IsendSize(shareSendSeqSize[I], I, sharingComm, shareSendSizeReq[I]);
      IsendSeq(shareSendSeq[I], I, sharingComm, shareSendSeqReq[I]);
    }
    completed.pop_back();
    // who's turn is next to receive an episode?
    if(sharingSize>0) sharingTurn = (sharingTurn+1) % sharingSize; //pick next
  }
}

void DataCoordinator::mastersRecvEpisodes()
{
  for(Uint i=0; i<sharingSize; ++i)
    if (i not_eq sharingRank) {
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
      if(shareSendSizeReq[i] not_eq MPI_REQUEST_NULL) {
        int complete = 0;
        MPI(Test, & shareSendSizeReq[i], &complete, MPI_STATUS_IGNORE);
        if(complete) assert(shareSendSizeReq[i] == MPI_REQUEST_NULL);
      }
      if( shareSendSeqReq[i] not_eq MPI_REQUEST_NULL) {
        int complete = 0;
        MPI(Test, &  shareSendSeqReq[i], &complete, MPI_STATUS_IGNORE);
        if(complete) assert(shareSendSizeReq[i] == MPI_REQUEST_NULL);
      }
    }

  assert(allTasksPtr not_eq nullptr);
  // if all learners are locking data acquisition we do not recv eps from worker
  // such that they wait for updated parameters before gathering more data
  if(allTasksPtr->dataAcquisitionIsLocked()) return;

  for(Uint i=1; i<workerSize; ++i)
    if (isComplete(workerRecvSizeReq[i]))
    {
      Fvec EP(workerRecvSeqSize[i], (Fval)0);
      const Uint nStep = Sequence::computeTotalEpisodeNstep(sI.dimObs(),
                                                      aI.dim(), aI.dimPol(),
                                                      workerRecvSeqSize[i]);
      RecvSeq(EP, i, workerComm);
      int bSendParams = 0;
      MPI(Recv, &bSendParams, 1, MPI_INT, i, 89, workerComm, MPI_STATUS_IGNORE);
      if(bSendParams) params.send(i, 0);

      nSeenTransitions_loc += nStep - 1; // we do not count init state
      nSeenSequences_loc += 1;
      // prepare the next one:
      IrecvSize(workerRecvSeqSize[i], i, workerComm, workerRecvSizeReq[i]);

      //_warn("sending to rank %lu", sharingTurn);
      if (sharingTurn == sharingRank) {
        Sequence * const tmp = new Sequence();
        tmp->unpackSequence(EP, sI.dimObs(), aI.dim(), aI.dimPol());
        assert(nStep == tmp->ndata() + 1);
        //_warn("storing new sequence of size %lu", tmp->ndata());
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
      if(sharingSize>0) sharingTurn = (sharingTurn+1) % sharingSize; //pick next
    }
}

// called externally
void DataCoordinator::addComplete(Sequence* const EP, const bool bUpdateParams)
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
    #ifndef NDEBUG
      Sequence * const tmp = new Sequence();
      tmp->unpackSequence(sendSq, sI.dimObs(), aI.dim(), aI.dimPol());
      //_warn("storing new sequence of size %lu", tmp->ndata());
      assert(EP->isEqual(tmp));
      delete tmp;
    #endif
    unsigned long sendSz = sendSq.size();
    MPI(Send, &sendSz, 1, MPI_UNSIGNED_LONG, 0, 99, workerComm);
    MPI(Send, sendSq.data(), sendSz, MPI_Fval, 0, 98, workerComm);
    const int intUpdateParams = bUpdateParams;
    MPI(Send, &intUpdateParams, 1, MPI_INT, 0, 89, workerComm);
    if(bUpdateParams) {
      //_warn("sent episode of size %lu and update params", EP->ndata() );
      params.recv(0);
    } //else _warn("sent ep of size %lu w/o update params", EP->ndata() );
    delete EP;
  }
  else // data stays here
  {
    replay->pushBackSequence(EP);
  }
}

}
