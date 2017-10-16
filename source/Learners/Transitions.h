/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "../StateAction.h"
#include "../Settings.h"
#include "../Environments/Environment.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>

struct Tuple
{
  vector<Real> s, a, mu;
  Real r = 0, SquaredError = 0;
  #ifdef importanceSampling
    Real weight = 0;
  #endif
};

struct Sequence
{
  vector<Tuple*> tuples;
  inline Uint ndata() const
  {
    assert(tuples.size());
    if(tuples.size()==0) return 0;
    return tuples.size()-1;
  }
  bool ended = false;
  Real MSE = 0;
  Uint ID = 0;
  ~Sequence()
  {
    for (auto & trash : tuples) _dispose_object( trash);
  }
};

struct Gen
{
  mt19937 * g;
  Gen(mt19937 * gen) : g(gen) { }
  size_t operator()(size_t n)
  {
    std::uniform_int_distribution<size_t> d(0, n ? n-1 : 0);
    return d(*g);
  }
};

class Transitions
{
public:
  const MPI_Comm mastersComm;
  Environment * const env;
  const bool bNormalize, bTrain, bWriteToFile, bSampleSeq;
  const Uint maxTotSeqNum, maxSeqLen, minSeqLen, nAppended, batchSize;
  const int learn_rank, learn_size;
  const string path;
  const Real gamma;
  bool first_pass = true;
  std::vector<std::mt19937>& generators;
  vector<Real> std, mean, invstd;
  vector<Uint> inds;
  discrete_distribution<Uint> * dist = nullptr;
  //bool bRecurrent;
  const StateInfo sI;
  const ActionInfo aI;
  const vector<Agent*> _agents;
  Uint anneal=0, nBroken=0, nTransitions=0, nSequences=0, old_ndata=0;
  size_t nSeenSequences=0, nSeenTransitions=0;
  Uint iOldestSaved = 0, printCount = 0;
  Uint adapt_TotSeqNum = maxTotSeqNum;
  Real invstd_reward = 1, mean_reward = 0;
  Gen * gen;
  vector<Sequence*> Set, Tmp, Buffered;
  mutable std::mutex dataset_mutex;

protected:
  int add(const int agentId, const Agent&a, const vector<Real>mu);
  void push_back(const int & agentId);
  void clear(const int & agentId);
  void synchronize();
  void sortSequences();

public:
  Transitions(MPI_Comm comm, Environment*const env, Settings & settings);

  ~Transitions()
  {
    _dispose_object(gen);
    _dispose_object(dist);
    for (auto & trash : Set) _dispose_object( trash);
    for (auto & trash : Tmp) _dispose_object( trash);
    for (auto & trash : Buffered) _dispose_object( trash);
  }
  void clearFailedSim(const int agentOne, const int agentEnd);
  void pushBackEndedSim(const int agentOne, const int agentEnd);
  //void update_samples_mean(const Real alpha = 0.01);

  //vector<Real> standardize(const vector<Real>& state, const Real noise=-1, const Uint thrID=0) const;
  template<typename T>
  inline vector<Real> standardize(const vector<T>& state) const
  {
    //if(!bNormalize) return state;
    vector<Real> tmp(sI.dimUsed*(1+nAppended));
    assert(state.size() == sI.dimUsed*(1+nAppended));
    for (Uint j=0; j<1+nAppended; j++)
      for (Uint i=0; i<sI.dimUsed; i++)
        tmp[i + j*sI.dimUsed] = (state[i + j*sI.dimUsed] - mean[i])*invstd[i];
    /*
      if (noise>0) {
        assert(generators.size()>thrID);
        std::normal_distribution<Real> distn(0.,noise);
        for (Uint i=0; i<sI.dimUsed*(1+nAppended); i++)
          tmp[i] += distn(generators[thrID]);
      }
    */
    return tmp;
  }
  inline vector<Real> standardized(const Uint seq, const Uint samp) const
  {
    return standardize(Set[seq]->tuples[samp]->s);
  }

  void update_rewards_mean();
  inline Real standardized_reward(const Uint seq, const Uint samp) const
  {
    assert(samp>0 && samp < Set[seq]->tuples.size());
    if(!bNormalize) return Set[seq]->tuples[samp]->r;
    return Set[seq]->tuples[samp]->r * invstd_reward;
    //return 1-gamma +(Set[seq]->tuples[samp]->r - mean_reward) * invstd_reward;
  }

#ifdef importanceSampling
  void updateP();
#endif
  void save(std::string fname);
  void restart(std::string fname);
  Uint updateSamples(const Real annealFac);
  int sample(const int thrID = 0);
  Uint restartSamples(const Uint polDim = 0);
  void saveSamples();

  int passData(const int agentId, const Agent&a, const vector<Real>mu = vector<Real>());
  void writeData(const int agentId, const Agent&a, const vector<Real>mu) const;

  Uint prune(const Real maxFrac, const Real CmaxRho);

  inline bool requestUpdateSamples() const
  {
    //three cases:
    // if i do not have enough shuffled indices for training a batchSize
    // if i have buffered transitions to add to dataset
    // if my desired dataset size is less than what i have
    return inds.size()<batchSize || Buffered.size() || adapt_TotSeqNum<Set.size();
  }

  inline Uint readNTransitions() const
  {
    //lock_guard<mutex> lock(dataset_mutex);
    return nTransitions;
  }
  inline void indexToSample(const int nSample, Uint& sequence, Uint& transition) const
  {
    int k = 0, back = 0, indT = Set[0]->ndata();
    while (nSample >= indT) {
      //printf("%u %u %u %u\n",k,back,indT,newSample);
      assert(k+2<=(int)Set.size());
      back = indT;
      indT += Set[++k]->ndata();
    }
    assert(nSample>=back);
    assert(Set[k]->ndata() > (Uint)nSample-back);
    sequence = k;
    transition = nSample-back;
  }
  inline Real readAvgSeqLen() const
  {
    //lock_guard<mutex> lock(dataset_mutex);
    return nTransitions/(nSequences+2.2e-16);
  }
  inline Uint readNSeen() const
  {
    //lock_guard<mutex> lock(dataset_mutex);
    #ifdef PACE_SEQUENCES
    return nSeenSequences;
    #else
    return nSeenTransitions;
    #endif
  }
  inline Uint readNData() const
  {
    //lock_guard<mutex> lock(dataset_mutex);
    #ifdef PACE_SEQUENCES
    return nSequences;
    #else
    return nTransitions;
    #endif
  }

  inline void popBackSequence()
  {
    removeSequence(nSequences-1);
    Set.pop_back();
    nSequences--;
    assert(nSequences==Set.size());
  }
  inline void pushBackSequence(Sequence*const seq)
  {
    assert(Set.size() < adapt_TotSeqNum);
    Set.push_back(nullptr);
    addSequence(nSequences, seq);
    nSequences++;
    assert(nSequences == Set.size());
  }
  inline void addSequence(const Uint ind, Sequence*const seq)
  {
    assert(Set[ind] == nullptr && seq not_eq nullptr);
    if (not seq->ended) ++nBroken;
    nTransitions += seq->ndata();
    Set[ind] = seq;
  }
  inline void removeSequence(const Uint ind)
  {
    assert(Set[ind] not_eq nullptr);
    if(not Set[ind]->ended) {
      assert(nBroken>0);
      --nBroken;
    }
    nTransitions -= Set[ind]->ndata();
    _dispose_object(Set[ind]);
    Set[ind] = nullptr;
  }
};
