//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MemoryBuffer_h
#define smarties_MemoryBuffer_h

#include "Sampling.h"
#include "Sequences.h"
#include "../Core/Agent.h"
#include "../Utils/Settings.h"
#include <atomic>
#include <mutex>

namespace smarties
{

class MemoryBuffer
{
 public:
  const MDPdescriptor & MDP;
  const Settings & settings;
  const DistributionInfo & distrib;
  const StateInfo sI = StateInfo(MDP);
  const ActionInfo aI = ActionInfo(MDP);
  Uint learnID = 0;

 private:

  friend class Sampling;
  friend class Collector;
  friend class MemorySharing;
  friend class MemoryProcessing;

  std::vector<std::mt19937>& generators = distrib.generators;
  std::vector<nnReal>& invstd = MDP.stateScale;
  std::vector<nnReal>& mean = MDP.stateMean;
  std::vector<nnReal>& std = MDP.stateStdDev;
  Real& stddev_reward = MDP.rewardsStdDev;
  Real& invstd_reward = MDP.rewardsScale;
  const Uint nAppended = MDP.nAppendedObs;

  const Uint dimS = MDP.dimState;
  const Real gamma = settings.gamma;

  std::atomic<bool> needs_pass {false};

  std::vector<Sequence*> Set;
  std::vector<Uint> sampled;
  std::mutex dataset_mutex;

  std::atomic<long> nSequences{0};
  std::atomic<long> nTransitions{0};
  std::atomic<long> nSeenSequences{0};
  std::atomic<long> nSeenTransitions{0};
  std::atomic<long> nSeenSequences_loc{0};
  std::atomic<long> nSeenTransitions_loc{0};

  const std::unique_ptr<Sampling> sampler;

  Real minPriorityImpW = 1;
  Real maxPriorityImpW = 1;

  void checkNData();

 public:

  MemoryBuffer(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_);
  ~MemoryBuffer();

  void initialize();

  void clearAll();

  template<typename T>
  Rvec standardizeAppended(const std::vector<T>& state) const {
    Rvec ret(sI.dimObs()*(1+nAppended));
    assert(state.size() == sI.dimObs()*(1+nAppended));
    for (Uint j=0, k=0; j<1+nAppended; ++j)
      for (Uint i=0; i<sI.dimObs(); ++i, ++k)
        ret[k] = (state[k]-mean[i]) * invstd[i];
    return ret;
  }
  template<typename T>
  Rvec standardize(const std::vector<T>& state) const {
    Rvec ret(sI.dimObs());
    assert(state.size() == sI.dimObs() && mean.size() == sI.dimObs());
    for (Uint i=0; i<sI.dimObs(); i++) ret[i] = (state[i]-mean[i])*invstd[i];
    return ret;
  }

  Real scaledReward(const Uint seq, const Uint samp) const {
    return scaledReward(Set[seq], samp);
  }
  Real scaledReward(const Sequence*const seq,const Uint samp) const {
    assert(samp < seq->rewards.size());
    return scaledReward(seq->rewards[samp]);
  }
  Real scaledReward(const Real r) const { return r * invstd_reward; }

  void restart(const std::string base);
  void save(const std::string base, const Uint nStep, const bool bBackup);

  void sample(std::vector<Uint>& seq, std::vector<Uint>& obs);

  long readNSeen_loc()    const { return nSeenTransitions_loc.load();  }
  long readNSeenSeq_loc() const { return nSeenSequences_loc.load();  }
  long readNData()        const { return nTransitions.load();  }
  long readNSeq()         const { return nSequences.load();  }
  void setNSeen_loc(const long val)    { nSeenTransitions_loc = val;  }
  void setNSeenSeq_loc(const long val) { nSeenSequences_loc = val;  }
  void setNData(const long val)        { nTransitions = val;  }
  void setNSeq(const long val) { nSequences = val; Set.resize(val, nullptr); }

  void popBackSequence();
  void removeSequence(const Uint ind);
  void pushBackSequence(Sequence*const seq);

  Sequence* get(const Uint ID) {
    return Set[ID];
  }
  void set(Sequence*const S, const Uint ID) {
    assert(Set[ID] == nullptr);
    Set[ID] = S;
  }

  float getMinPriorityImpW() { return minPriorityImpW; }
  float getMaxPriorityImpW() { return maxPriorityImpW; }
  const bool requireImpWeights = sampler->requireImportanceWeights();

  const std::vector<Uint>& listSampled() { return sampled; }
  static Sampling* prepareSampler(const Settings&S, MemoryBuffer* const R);
};

}
#endif
