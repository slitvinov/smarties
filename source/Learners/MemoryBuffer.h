/*
 *  QApproximator.h
 *  rl
 *
 *  Created by Guido Novati on 24.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "Sequences.h"
#include "../Environments/Environment.h"
#include <parallel/algorithm>

enum FORGET {OLDEST, MAXERROR, MINERROR};
class MemoryBuffer
{
public:
  const MPI_Comm mastersComm;
  Environment * const env;
  const bool bWriteToFile, bTrain, bSampleSeq;
  const Uint nAppended, batchSize, maxTotObsNum, nThreads, policyVecDim;
  const StateInfo sI;
  const ActionInfo aI;
  const vector<Agent*> _agents;
  std::vector<std::mt19937>& generators;
  const Rvec mean, invstd, std;
  const int learn_rank, learn_size;
  const Real gamma;

  bool first_pass = true;
  discrete_distribution<Uint> * dist = nullptr;
  //bool bRecurrent;
  Uint nBroken=0, nTransitions=0, nSequences=0;
  Uint nTransitionsInBuf=0, nTransitionsDeleted=0;
  size_t nSeenSequences=0, nSeenTransitions=0, iOldestSaved = 0;
  Uint nPruned = 0, minInd = 0;
  Real invstd_reward = 1, mean_reward = 0, nOffPol = 0, totMSE = 0;


  Gen* gen;
  vector<Sequence*> Set, inProgress;
  mutable std::mutex dataset_mutex;

  MPI_Request rewRequest = MPI_REQUEST_NULL;
  long double rew_reduce_result[2], partial_sum[2];

public:
  void push_back(const int & agentId);

  MemoryBuffer(Environment*const env, Settings & settings);

  ~MemoryBuffer()
  {
    _dispose_object(gen);
    _dispose_object(dist);
    for (auto & trash : Set) _dispose_object( trash);
    for (auto & trash : inProgress) _dispose_object( trash);
  }

  void inline clearAll()
  {
    for(auto& old_traj: Set) //delete already-used trajectories
      _dispose_object(old_traj);
    //for(auto& old_traj: data->inProgress)
    //  old_traj->clear();//remove from in progress: now off policy
    Set.clear(); //clear trajectories used for learning
    nBroken = 0;
    nSequences = 0;
    nTransitions = 0;
  }

  template<typename T>
  inline Rvec standardize(const vector<T>& state) const
  {
    Rvec ret(sI.dimUsed*(1+nAppended));
    assert(state.size() == sI.dimUsed*(1+nAppended));
    for (Uint j=0; j<1+nAppended; j++)
      for (Uint i=0; i<sI.dimUsed; i++)
        ret[j +i*(nAppended+1)] =(state[j +i*(nAppended+1)]-mean[i])*invstd[i];
    return ret;
  }
  inline Rvec standardized(const Uint seq, const Uint samp) const
  {
    return standardize(Set[seq]->tuples[samp]->s);
  }
  inline Real standardized_reward(const Uint seq, const Uint samp) const
  {
    assert(samp>0 && samp < Set[seq]->tuples.size());
    return Set[seq]->tuples[samp]->r * invstd_reward;
  }
  inline Real standardized_reward(const Sequence*const seq,const Uint samp)const
  {
    assert(samp < seq->tuples.size()); // samp>0 &&
    return seq->tuples[samp]->r * invstd_reward;
  }

  void clearFailedSim(const int agentOne, const int agentEnd)
  {
    for (int i = agentOne; i<agentEnd; i++) {
      _dispose_object(inProgress[i]);
      inProgress[i] = new Sequence();
    }
  }
  void pushBackEndedSim(const int agentOne, const int agentEnd)
  {
    for(int i=agentOne; i<agentEnd; i++) if(inProgress[i]->ndata()) push_back(i);
  }

  void add_action(const Agent& a, Rvec pol = Rvec()) const;
  void terminate_seq(const Agent&a);
  int add_state(const Agent&a);

  void updateRewardsStats();
  void updateImportanceWeights();
  void prune(const Real CmaxRho, const FORGET ALGO);

  void getMetrics(ostringstream& buff);
  void getHeaders(ostringstream& buff);
  void restart();

  void indexToSample(const int nSample, Uint& seq, Uint& obs) const;
  void sampleTransition(Uint& seq, Uint& obs, const int thrID);
  Uint sampleTransition(const Uint seq, const int thrID);
  void sampleSequence(Uint& seq, const int thrID);
  vector<Uint> sampleSequences(const Uint N);

  inline Uint readNTransitions() const
  {
    //lock_guard<mutex> lock(dataset_mutex);
    return nTransitions;
  }
  inline Real readAvgSeqLen() const
  {
    //lock_guard<mutex> lock(dataset_mutex);
    return nTransitions/(nSequences+2.2e-16);
  }
  inline Uint readNSeen() const
  {
    //lock_guard<mutex> lock(dataset_mutex);
    //#ifdef PACE_SEQUENCES
    //return nSeenSequences;
    //#else
    return nSeenTransitions;
    //#endif
  }
  inline Uint readNData() const
  {
    //lock_guard<mutex> lock(dataset_mutex);
    //#ifdef PACE_SEQUENCES
    //return nSequences;
    //#else
    return nTransitions;
    //#endif
  }

  inline void popBackSequence()
  {
    removeSequence(nSequences-1);
    Set.pop_back();
    #pragma omp atomic
    nSequences--;
    assert(nSequences==Set.size());
  }
  inline void pushBackSequence(Sequence*const seq)
  {
    lock_guard<mutex> lock(dataset_mutex);
    Set.push_back(nullptr);
    addSequence(nSequences, seq);
    #pragma omp atomic
    nSequences++;
    assert(nSequences == Set.size());
  }
  inline void addSequence(const Uint ind, Sequence*const seq)
  {
    assert(Set[ind] == nullptr && seq not_eq nullptr);
    if (not seq->ended) ++nBroken;
    #pragma omp atomic
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
    #pragma omp atomic
    nTransitions -= Set[ind]->ndata();
    _dispose_object(Set[ind]);
    Set[ind] = nullptr;
  }

  inline void checkNData()
  {
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
