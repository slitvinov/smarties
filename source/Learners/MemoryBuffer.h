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
#include <parallel/algorithm>

enum FORGET {OLDEST, FARPOLFRAC, MAXKLDIV, MINERROR};
class MemoryBuffer
{
public:
  Settings & settings;
  const MPI_Comm mastersComm = settings.mastersComm;
  Environment * const env;
  const bool bWriteToFile = settings.samplesFile, bTrain = settings.bTrain;
  const bool bSampleSeq = settings.bSampleSequences;
  const Uint nAppended = settings.appendedObs, batchSize = settings.batchSize;
  const Uint maxTotObsNum = settings.maxTotObsNum;
  const Uint nThreads = settings.nThreads, policyVecDim = settings.policyVecDim;
  const StateInfo& sI = env->sI;
  const ActionInfo& aI = env->aI;
  const vector<Agent*>& _agents = env->agents;
  std::vector<std::mt19937>& generators = settings.generators;

  vector<memReal> invstd = sI.inUseInvStd();
  vector<memReal> mean = sI.inUseMean();
  vector<memReal> std = sI.inUseStd();
  Real invstd_reward = 1;

  const Real gamma = settings.gamma;
  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;
  const int ESpopSize = settings.ESpopSize;
  bool needs_pass = true;
  #ifdef PRIORITIZED_ER
    Uint stepSinceISWeep = 0;
    discrete_distribution<Uint> distPER;
    float minPriorityImpW = 1;
    float maxPriorityImpW = 1;
    void updateImportanceWeights();
  #endif

  Uint nPruned = 0, minInd = 0, learnID = 0;
  Real nOffPol = 0, avgDKL = 0;
  int del_ptr = -1;

  vector<Sequence*> Set, inProgress;
  mutable std::mutex dataset_mutex;

  const Uint dimS = sI.dimUsed, nReduce = 2 + 2*dimS;
  ApproximateReductor reductor = ApproximateReductor(mastersComm, nReduce);

  std::atomic<Uint> nTransitions{0}, nSequences{0};
  std::atomic<Uint> nSeenSequences{0}, nSeenTransitions{0};
  std::atomic<Uint> nCmplTransitions{0}, iOldestSaved{0};
public:
  void push_back(const int & agentId);

  MemoryBuffer(Environment*const env, Settings & settings);

  ~MemoryBuffer()
  {
    for (auto & trash : Set) _dispose_object( trash);
    for (auto & trash : inProgress) _dispose_object( trash);
  }

  void initialize()
  {
    // All sequences obtained before this point should share the same time stamp
    for(Uint i=0;i<Set.size();i++) Set[i]->ID = readNSeenSeq()-1;
    #ifdef PRIORITIZED_ER
    vector<float> probs(nTransitions.load(), 1);
    distPER = discrete_distribution<Uint>(probs.begin(), probs.end());
    #endif
  }

  void inline clearAll()
  {
    for(auto& old_traj: Set) //delete already-used trajectories
      _dispose_object(old_traj);
    //for(auto& old_traj: data->inProgress)
    //  old_traj->clear();//remove from in progress: now off policy
    Set.clear(); //clear trajectories used for learning
    nSequences = 0;
    nTransitions = 0;
  }

  // TODO PREFIX: FORCE RECOUNT AFTER THIS:
  // Uint clearOffPol(const Real C, const Real tol)
  // {
  //   Uint i = 0;
  //   while(1) {
  //     if(i>=Set.size()) break;
  //     Uint _nOffPol = 0;
  //     for(Uint j=0; j<Set[i]->ndata(); j++)
  //       _nOffPol +=(Set[i]->offPolicImpW[j]>1+C || Set[i]->offPolicImpW[j]<1-C);
  //     if(_nOffPol > tol*Set[i]->ndata()) {
  //       std::swap(Set[i], Set.back());
  //       popBackSequence();
  //     }
  //     else i++;
  //   }
  //   return readNData();
  // }

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

  #ifdef NOISY_INPUT
  inline Rvec standardizeNoisy(const Sequence*const traj, const int t,
      const Uint thrID) const {
    Rvec ret = standardize(traj->tuples[t]->s);
    const Rvec nxt = standardize(traj->tuples[traj->isLast(t+1) ? t : t+1]->s);
    const Rvec prv = standardize(traj->tuples[t>0 ? t-1 : t]->s);
    std::normal_distribution<Real> noise(0, NOISY_INPUT);
    for (Uint i=0; i<sI.dimUsed; i++) {
      // i don't care about the sign: Gaussian is symmetric
      ret[i] += (nxt[i]-prv[i])*noise(generators[thrID]);
    }
    return ret;
  }
  #endif // NOISY_INPUT

  inline Real scaledReward(const Uint seq, const Uint samp) const {
    assert(samp>0 && samp < Set[seq]->tuples.size());
    return Set[seq]->tuples[samp]->r * invstd_reward;
  }
  inline Real scaledReward(const Sequence*const seq,const Uint samp)const
  {
    assert(samp < seq->tuples.size()); // samp>0 &&
    return seq->tuples[samp]->r * invstd_reward;
  }

  void add_action(const Agent& a, Rvec pol);
  void terminate_seq(Agent&a);
  void add_state(const Agent&a);

  void updateRewardsStats(unsigned long nStep, Real WR = 1, Real WS = -1);

  static FORGET readERfilterAlgo(const string setting, const bool bReFER)
  {
    if(setting == "oldest")     return OLDEST;
    if(setting == "farpolfrac") return FARPOLFRAC;
    if(setting == "maxkldiv")   return MAXKLDIV;
    if(setting == "minerror")   return MINERROR;
    if(setting == "default") {
      if(bReFER) return FARPOLFRAC;
      else       return OLDEST;
    }
    die("ERoldSeqFilter not recognized");
    return OLDEST; // to silence warning
  }
  // Algorithm for maintaining and filtering dataset, and optional imp weight range parameter
  void prune(const FORGET ALGO, const Fval CmaxRho = 1);
  void finalize();

  void getMetrics(ostringstream& buff);
  void getHeaders(ostringstream& buff);
  void restart(const string base);
  void save(const string base, const Uint nStep, const bool bBackup);

  void sampleTransitions(vector<Uint>& seq, vector<Uint>& obs);
  void sampleSequences(vector<Uint>& seq);

  inline Uint readNSeen() const {
    return nSeenTransitions.load();
  }
  inline Uint readNConcluded() const {
    return nCmplTransitions.load();
  }
  inline Uint readNSeenSeq() const {
    return nSeenSequences.load();
  }
  inline Uint readNData() const {
    return nTransitions.load();
  }
  inline Uint readNSeq() const {
    return nSequences.load();
  }

 private:

  void prefixSum();
  void popBackSequence();
  void pushBackSequence(Sequence*const seq);

  inline void checkNData() {
    #ifndef NDEBUG
      Uint cntSamp = 0;
      for(Uint i=0; i<Set.size(); i++) {
        assert(Set[i] not_eq nullptr);
        cntSamp += Set[i]->ndata();
      }
      assert(cntSamp==nTransitions);
      assert(Set.size()==nSequences);
    #endif
  }
};
