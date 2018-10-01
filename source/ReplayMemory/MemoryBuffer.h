//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "Sequences.h"
#include <atomic>
#include "../Environments/Environment.h"
#include "StatsTracker.h"

class MemoryBuffer
{
 public:
  const Settings & settings;
  const Environment * const env;

  const StateInfo& sI = env->sI;
  const ActionInfo& aI = env->aI;
  const vector<Agent*>& agents = env->agents;
  Uint learnID = 0;
 private:

  friend class Collector;
  friend class MemorySharing;
  friend class MemoryProcessing;

  const Uint nAppended = settings.appendedObs;
  std::vector<std::mt19937>& generators = settings.generators;

  std::vector<memReal> invstd = sI.inUseInvStd();
  std::vector<memReal> mean = sI.inUseMean();
  std::vector<memReal> std = sI.inUseStd();
  Real invstd_reward = 1;

  const Uint dimS = mean.size();
  const Real gamma = settings.gamma;
  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;

  std::atomic<bool> needs_pass {true};

  std::vector<Sequence*> Set;

  std::mutex dataset_mutex;

  std::atomic<Uint> nSequences{0};
  std::atomic<Uint> nTransitions{0};
  std::atomic<Uint> nSeenTransitions{0};
  std::atomic<Uint> nSeenSequences_loc{0};
  std::atomic<Uint> nSeenTransitions_loc{0};

  void checkNData();

 public:

  MemoryBuffer(const Settings& settings, const Environment*const env);
  ~MemoryBuffer();

  void initialize();

  void clearAll();

  template<typename T>
  inline Rvec standardizeAppended(const vector<T>& state) const {
    Rvec ret(sI.dimUsed*(1+nAppended));
    assert(state.size() == sI.dimUsed*(1+nAppended));
    for (Uint j=0; j<1+nAppended; j++)
      for (Uint i=0; i<sI.dimUsed; i++)
        ret[j +i*(nAppended+1)] =(state[j +i*(nAppended+1)]-mean[i])*invstd[i];
    return ret;
  }
  template<typename T>
  inline Rvec standardize(const vector<T>& state) const {
    Rvec ret(sI.dimUsed);
    assert(state.size() == sI.dimUsed && mean.size() == sI.dimUsed);
    for (Uint i=0; i<sI.dimUsed; i++) ret[i] =(state[i]-mean[i])*invstd[i];
    return ret;
  }

  inline Real scaledReward(const Uint seq, const Uint samp) const {
    return scaledReward(Set[seq], samp);
  }
  inline Real scaledReward(const Sequence*const seq,const Uint samp) const {
    assert(samp < seq->tuples.size());
    return scaledReward(seq->tuples[samp]->r);
  }
  inline Real scaledReward(const Real r) const { return r * invstd_reward; }

  void restart(const string base);
  void save(const string base, const Uint nStep, const bool bBackup);

  void sampleTransitions(vector<Uint>& seq, vector<Uint>& obs);
  void sampleSequences(vector<Uint>& seq);

  inline Uint readNSeen_loc() const {
    return nSeenTransitions_loc.load();
  }
  inline Uint readNSeenSeq_loc() const {
    return nSeenSequences_loc.load();
  }
  inline Uint readNSeen() const {
    return nSeenTransitions.load();
  }
  inline Uint readNData() const {
    return nTransitions.load();
  }
  inline Uint readNSeq() const {
    return nSequences.load();
  }

  inline void setNSeen_loc(const Uint val) {
    nSeenTransitions_loc = val;
  }
  inline void setNSeenSeq_loc(const Uint val) {
    nSeenSequences_loc = val;
  }
  inline void setNSeen(const Uint val) {
    nSeenTransitions = val;
  }
  inline void setNData(const Uint val) {
    nTransitions = val;
  }
  inline void setNSeq(const Uint val) {
    nSequences = val;
    Set.resize(val, nullptr);
  }

  void popBackSequence();
  void removeSequence(const Uint ind);
  void pushBackSequence(Sequence*const seq);

  inline Sequence* get(const Uint ID) {
    return Set[ID];
  }
  inline void set(Sequence*const S, const Uint ID) {
    assert(Set[ID] == nullptr);
    Set[ID] = S;
  }
};
