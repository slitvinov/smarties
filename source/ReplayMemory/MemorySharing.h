//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MemorySharing_h
#define smarties_MemorySharing_h

#include "MemoryBuffer.h"
#include <thread>

namespace smarties
{

struct MemorySharing
{
  MemoryBuffer* const replay;
  const Settings & settings = replay->settings;
  const DistributionInfo & distrib = replay->distrib;

  const StateInfo& sI = replay->sI;
  const ActionInfo& aI = replay->aI;
  std::vector<Sequence*> completed;

  // allows masters to share episodes between each others
  // each master sends the size (in floats) of the episode
  // then sends the episode itself. same goes for receiving
  MPI_Comm sharingComm = MPI_COMM_NULL;
  Uint sharingSize, sharingRank, sharingTurn;
  std::vector<MPI_Request> shareSendSizeReq, shareSendSeqReq, shareRecvSizeReq;
  std::vector<unsigned long> shareSendSeqSize, shareRecvSeqSize;
  std::vector<Fvec> shareSendSeq;

  MPI_Comm workerComm = MPI_COMM_NULL;
  Uint workerSize, workerRank;
  std::vector<MPI_Request> workerRecvSizeReq;
  std::vector<unsigned long> workerRecvSeqSize;

  std::mutex complete_mutex;

  std::thread fetcher;
  std::atomic<Uint> bFetcherRunning {0};

  std::atomic<long>& nSeenTransitions_loc = replay->nSeenTransitions_loc;
  std::atomic<long>& nSeenSequences_loc = replay->nSeenSequences_loc;
  long int globSeen[2] = {0, 0};

  MemorySharing(MemoryBuffer*const RM);
  ~MemorySharing();

  inline int testBuffer(MPI_Request& req);

  void recvEP(const int ID2);

  void sendEp(const int ID2, Sequence* const EP);

  void run();

  void addComplete(Sequence* const EP);

  void IrecvSize(unsigned long& size, const int rank, const MPI_Comm C, MPI_Request&R) const
  {
    MPI(Irecv, &size, 1, MPI_UNSIGNED_LONG, rank, 99, C, &R);
  }
  void IsendSize(const unsigned long& size, const int rank, const MPI_Comm C, MPI_Request&R) const
  {
    MPI(Isend, &size, 1, MPI_UNSIGNED_LONG, rank, 99, C, &R);
  }

  void RecvSeq(Fvec&V, const int rank, const MPI_Comm C) const
  {
    MPI( Recv, V.data(), V.size(), MPI_Fval, rank, 98, C, MPI_STATUS_IGNORE);
  }
  void IsendSeq(const Fvec&V, const int rank, const MPI_Comm C, MPI_Request&R) const
  {
    MPI(Isend, V.data(), V.size(), MPI_Fval, rank, 98, C, &R);
  }

  bool isComplete(MPI_Request& req)
  {
    if(req == MPI_REQUEST_NULL) return false;
    int bRecvd = 0;
    MPI(Test, &req, &bRecvd, MPI_STATUS_IGNORE);
    return bRecvd > 0;
  }
};

}
#endif
