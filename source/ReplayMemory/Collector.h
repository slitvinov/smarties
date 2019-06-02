//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Collector_h
#define smarties_Collector_h

#include "MemorySharing.h"
#include "Utils/StatsTracker.h"

namespace smarties
{

class Collector
{
private:
  MemoryBuffer * const replay;
  const std::unique_ptr<MemorySharing> sharing;
  const MDPdescriptor & MDP = replay->MDP;
  const Settings & settings = replay->settings;
  const DistributionInfo & distrib = replay->distrib;
  const StateInfo& sI = replay->sI;
  const ActionInfo& aI = replay->aI;

  std::vector<Sequence*> inProgress;

  DelayedReductor<long> globalStep_reduce;

  std::atomic<long>& nSeenSequences = replay->nSeenSequences;
  std::atomic<long>& nSeenTransitions = replay->nSeenTransitions;
  std::atomic<long>& nSeenSequences_loc = replay->nSeenSequences_loc;
  std::atomic<long>& nSeenTransitions_loc = replay->nSeenTransitions_loc;

public:
  void add_state(Agent&a);
  void add_action(const Agent& a, const Rvec pol);
  void terminate_seq(Agent&a);
  void push_back(const int & agentId);

  inline Sequence* get(const Uint ID) const {
    return inProgress[ID];
  }
  inline Uint nInProgress() const {
    return inProgress.size();
  }

  Collector(MemoryBuffer*const RM);

  ~Collector();
};

}
#endif
