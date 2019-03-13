//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemorySharing.h"

MemorySharing::MemorySharing(const Settings&S, MemoryBuffer*const RM) : settings(S), replay(RM)
{
  completed.reserve(S.nAgents);
  if(SZ <= 1) return;
  #pragma omp parallel
  {
    const int thrID = omp_get_thread_num();
    const int tgtCPU =  std::min(omp_get_num_threads()-1, 2);
    if( thrID==tgtCPU ) fetcher = std::thread( [ &, this ] () { run(); } );
  }
}

MemorySharing::~MemorySharing()
{
  bExit = 1;
  if(SZ > 1) fetcher.join();
  for(const auto& S : completed) _dispose_object(S);
}

int MemorySharing::testBuffer(MPI_Request& req)
{
  if(req == MPI_REQUEST_NULL) return 0;
  int bRecvd = 0;
  MPI(Test, &req, &bRecvd, MPI_STATUS_IGNORE);
  return bRecvd;
}

void MemorySharing::recvEP(const int ID2)
{
  if( testBuffer(RRq[ID2]) ) {
   recvSq[ID2].resize(recvSz[ID2]);
   const auto NOSTS = MPI_STATUS_IGNORE;

   MPI(Recv, recvSq[ID2].data(), recvSz[ID2], MPI_Fval, ID2, 98, comm, NOSTS);
   MPI(Irecv, &recvSz[ID2], 1, MPI_UNSIGNED, ID2, 99, comm, &RRq[ID2]);

   Sequence * const tmp = new Sequence();
   tmp->unpackSequence(recvSq[ID2], dimS, dimA, dimP);
   replay->pushBackSequence(tmp);
  }
}

void MemorySharing::sendEp(const int ID2, Sequence* const EP)
{
  if(CRq[ID2] not_eq MPI_REQUEST_NULL) MPI(Wait, &CRq[ID2], MPI_STATUS_IGNORE);
  if(SRq[ID2] not_eq MPI_REQUEST_NULL) MPI(Wait, &SRq[ID2], MPI_STATUS_IGNORE);

  sendSq[ID2] = EP->packSequence(dimS, dimA, dimP);
  sendSz[ID2] = sendSq[ID2].size();

  MPI(Isend, &sendSz[ID2], 1, MPI_UNSIGNED, ID2, 99, comm, &CRq[ID2]);
  MPI(Isend, sendSq[ID2].data(),sendSz[ID2], MPI_Fval,ID2,98,comm,&SRq[ID2]);
}

void MemorySharing::run()
{
  for(int i=0; i<SZ; i++) if(i not_eq ID)
    MPI(Irecv, &recvSz[i], 1, MPI_UNSIGNED, i, 99, comm, &RRq[i]);

  while(true)
  {
    {
      std::lock_guard<std::mutex> lock(complete_mutex);
      while ( completed.size() ) {
        if (EpOwnerID == ID) replay->pushBackSequence(completed.back());
        else {
          sendEp(EpOwnerID, completed.back());
          _dispose_object(completed.back());
        }
        completed.pop_back();
        EpOwnerID = (EpOwnerID+1) % SZ;
      }
    }
    for(int i=0; i<SZ; i++) if(i not_eq ID) recvEP(i);

    usleep(1);
    if( bExit.load() > 0 ) break;
  }
}

void MemorySharing::addComplete(Sequence* const EP)
{
  if(SZ <= 1) {
    replay->pushBackSequence(EP);
  } else {
    std::lock_guard<std::mutex> lock(complete_mutex);
    completed.push_back(EP);
  }
}
