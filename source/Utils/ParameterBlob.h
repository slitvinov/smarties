//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_ParameterBlob_h
#define smarties_ParameterBlob_h

#include "Warnings.h"
#include "../Settings.h"
#include <utility>
#include <vector>
#include <atomic>
#include <cstring>

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
  const MPI_Comm comm = distrib.master_workers_comm;
  std::vector<dataInfo> dataList;
  long & nDataGatheredB4Startup;
  std::atomic<long>& nGradSteps;
  Real& avgCumulativeRew;
  const Uint counterMsgSize = 2 * sizeof(long) + sizeof(Real);
  void * const buffer = malloc(counterMsgSize);
  long tmpnGrads;

public:
  ParameterBlob(const DistributionInfo& D, long& nDataB4, std::atomic<long>& nG, Real& avgR)
    : distrib(D), nDataGatheredB4Startup(nDataB4), nGradSteps(nG), avgCumulativeRew(avgR) {}

  ~ParameterBlob() { free(buffer); }

  void add(const Uint size, nnReal * const data) {
    dataList.emplace_back(std::make_pair(size, data));
  }

  void recv(const Uint MDP_ID) const
  {
    long nGrads;
    MPI(Recv, buffer, counterMsgSize, MPI_BYTE, 0, 72726+MDP_ID, comm, MPI_STATUS_IGNORE);
    memcpy(&nDataGatheredB4Startup, buffer, sizeof(long));
    memcpy(&nGrads,                 buffer, sizeof(long));
    memcpy(&avgCumulativeRew,       buffer, sizeof(Real));
    nGradSteps = nGrads;

    // workers always recv params from learner (rank 0)
    for(const auto& data : dataList ) {
      MPI(Recv, data.second, data.first, SMARTIES_MPI_NNVALUE_TYPE, 0,
        72726 + MDP_ID, comm, MPI_STATUS_IGNORE);
    }
  }

  void send(const Uint toRank, const Uint MDP_ID)
  {
    const long nGrads = nGradSteps.load();
    memcpy(buffer, &nDataGatheredB4Startup, sizeof(long));
    memcpy(buffer, &nGrads,                 sizeof(long));
    memcpy(buffer, &avgCumulativeRew,       sizeof(Real));
    MPI(Send, buffer, counterMsgSize, MPI_BYTE, toRank, 72726 + MDP_ID, comm);

    for(const auto& data : dataList )
      MPI(Send, data.second, data.first, SMARTIES_MPI_NNVALUE_TYPE, toRank,
        72726 + MDP_ID, comm);
  }
};

} // end namespace smarties
#endif // smarties_ParameterBlob_h
