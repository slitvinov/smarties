//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ParameterBlob_h
#define smarties_ParameterBlob_h

#include "Utils/Warnings.h"
#include "Settings.h"
#include <utility>
#include <vector>

namespace smarties
{

// Each part of the code where learner parameters need be sent
// from learning algorithm to agents/workers needs to pass
// pointer to this class.
// As of now, these parameters are only network parameters and
// scaling coefficient for state/rewards, therefore assumed
// ptr to nnReal. Might make it general into char in the future.
// Receiving should be blocking, because requested by user/app
// Sending should be issued when a learner cannot proceed with
// more training? Non-blocking?
// Cannot send parameters at the end of each episode because
// many agents may share weights... but that can be seen as
// a secondary optimization...

class ParameterBlob
{
  using dataInfo = std::pair<Uint, nnReal*>;
  const DistributionInfo& distrib;
  const MPI_Comm comm = MPICommDup (distrib.master_workers_comm);
  const Uint nWorkers = MPICommSize(comm);
  Uint lastCommGradID = 0; // shared initialization
  std::vector<MPI_Request> sendReqs;

  std::vector<dataInfo> dataList;

public:

  ParameterBlob(const DistributionInfo& D) : distrib(D) {}

  void add(const Uint size, nnReal * const data) {
    dataList.emplace_back(std::make_pair(size, data));
  }

  void recv(const Uint gradStepID) const
  {
    //if(gradStepID == lastCommGradID)
    //  die("Asked parameter update two times at same gradStep.");

    // workers always recv params from learner (rank 0)
    for(const auto& data : dataList ) {
      MPI(Recv, data.second, data.first, MPI_NNVALUE_TYPE, 0,
        72726, comm, MPI_STATUS_IGNORE);
    }
  }

  void send(const Uint toRank, const Uint gradStepID)
  {
    //if(gradStepID == lastCommGradID) return;
    Uint transfID = dataList.size() * toRank;
    const Uint endTransID = transfID + dataList.size();
    if(sendReqs.size() <= endTransID)
      sendReqs.resize(endTransID, MPI_REQUEST_NULL);

//    for(Uint i=1, k=0; i<nWorkers; ++i) { // 0 is master
    for(const auto& data : dataList )
    {
      if(sendReqs[transfID] not_eq MPI_REQUEST_NULL)
        MPI(Wait, & sendReqs[transfID], MPI_STATUS_IGNORE);

      //MPI(Isend, data.second, data.first, MPI_NNVALUE_TYPE, toRank,
      //  72726, comm, & sendReqs[transfID]);
      MPI(Send, data.second, data.first, MPI_NNVALUE_TYPE, toRank,
        72726, comm);

      transfID++;
    }
  }
};

} // end namespace smarties
#endif // smarties_ParameterBlob_h
