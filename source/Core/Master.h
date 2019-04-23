//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Master_h
#define smarties_Master_h

#include "Worker.h"
#include <thread>

namespace smarties
{

class MasterSockets : Master<MasterSockets>
{
  using Request_t = SOCKET_REQ;

protected:
  void Irecv(void*const buffer, const Uint size, const int rank,
    const int tag, Request_t& request) const {
    SOCKET_Irecv(buffer, size, getSocketID(workerID), request);
  }

  void  Send(void*const buffer, const Uint size, const int rank,
    const int tag) const {
    SOCKET_Bsend(buffer, size, getSocketID(worker));
  }

  int TestComm(Request_t& request) const {
    SOCKET_Test(request.completed, request);
    return request.completed;
  }
  void WaitComm(Request_t& request) const {
    SOCKET_Wait(request);
  }

public:
  MasterSockets(Settings& settings, DistributionInfo& distribinfo);
};

class MasterMPI : Master<MasterMPI>
{
  using Request_t = MPI_Request;

protected:

  void Irecv(void*const buffer, const Uint size, const int rank,
    const int tag, Request_t& req) const {
    MPI(Irecv, buffer, size, MPI_BYTE, rank, tag, master_workers_comm, & req);
  }

  void  Send(void*const buffer, const Uint size, const int rank,
    const int tag) const {
    MPI(Send, buffer, size, MPI_BYTE, rank, tag, master_workers_comm);
  }

  int TestComm(Request_t& request) const {
    int completed = 0; MPI_Status mpistatus;
    MPI(Test, &request, &completed, MPI_STATUS_IGNORE);
    return completed;
  }
  void WaitComm(Request_t& request) const {
    MPI_Status mpistatus;
    MPI(Wait, &request, MPI_STATUS_IGNORE);
  }

public:
  MasterMPI(Settings& settings, DistributionInfo& distribinfo);
};

template <typename CommType>
class Master : public Worker
{
protected:
  std::vector<std::thread> worker_replies;
  CommType * interface() { return static_cast<CommType*> (this); }

public:
  Master(Settings& settings, DistributionInfo& distribinfo);
};

}
#endif
