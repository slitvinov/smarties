//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "MemorySharing.h"

class Collector
{
private:
  const Settings & settings;
  const Environment * const env;
  MemoryBuffer * const replay;
  MemorySharing * const sharing;

  const bool bWriteToFile = settings.samplesFile;
  const Uint policyVecDim = settings.policyVecDim;
  const int learn_rank = settings.learner_rank;

  std::vector<Sequence*> inProgress;

  std::atomic<Uint>& nSeenSequences_loc = replay->nSeenSequences_loc;
  std::atomic<Uint>& nSeenTransitions_loc = replay->nSeenTransitions_loc;

public:
  void add_state(const Agent&a);
  void add_action(const Agent& a, const Rvec pol);
  void terminate_seq(Agent&a);
  void push_back(const int & agentId);
  inline Sequence* get(const Uint ID) {
    return inProgress[ID];
  }
  Collector(const Settings&S, Learner*const L, MemoryBuffer*const RM);

  ~Collector();
};
