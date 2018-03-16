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
  size_t nSeenSequences=0, nSeenTransitions=0, nCmplTransitions=0, iOldestSaved = 0;
  Uint nPruned = 0, minInd = 0, _nStep = 0;
  Real invstd_reward = 1, mean_reward = 0, nOffPol = 0, totMSE = 0;


  Gen* gen;
  vector<Sequence*> Set, inProgress;
  mutable std::mutex dataset_mutex;

  const Uint dimS = sI.dimUsed, nReduce = 2;
  MPI_Request rewRequest = MPI_REQUEST_NULL;
  vector<long double> rew_reduce_result, partial_sum;

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
  inline Rvec standardizeAppended(const vector<T>& state) const
  {
    Rvec ret(sI.dimUsed*(1+nAppended));
    assert(state.size() == sI.dimUsed*(1+nAppended));
    for (Uint j=0; j<1+nAppended; j++)
      for (Uint i=0; i<sI.dimUsed; i++)
        ret[j +i*(nAppended+1)] =(state[j +i*(nAppended+1)]-mean[i])*invstd[i];
    return ret;
  }
  template<typename T>
  inline Rvec standardize(const vector<T>& state) const
  {
    Rvec ret(sI.dimUsed);
    assert(state.size() == sI.dimUsed && mean.size() == sI.dimUsed);
    for (Uint i=0; i<sI.dimUsed; i++) ret[i] =(state[i]-mean[i])*invstd[i];
    return ret;
  }

  inline Rvec standardizeNoisy(const Sequence*const traj, const int t,
      const Uint thrID) const {
    Rvec ret = standardize(traj->tuples[t]->s);
    const Rvec nxt = standardize(traj->tuples[traj->isLast(t) ? t : t+1]->s);
    const Rvec prv = standardize(traj->tuples[t>0 ? t-1 : t]->s);
    std::normal_distribution<Real> noise(0, 0.01/(1 + _nStep * ANNEAL_RATE) );
    for (Uint i=0; i<sI.dimUsed; i++) {
      // i don't care about the sign: Gaussian is symmetric
      ret[i] += (nxt[i]-prv[i])*noise(generators[thrID]);
    }
    return ret;
  }

  inline Real scaledReward(const Uint seq, const Uint samp) const
  {
    assert(samp>0 && samp < Set[seq]->tuples.size());
    return Set[seq]->tuples[samp]->r * invstd_reward;
  }
  inline Real scaledReward(const Sequence*const seq,const Uint samp)const
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
  void add_state(const Agent&a);

  void updateRewardsStats(unsigned long nStep);
  void updateImportanceWeights();
  void prune(const Real CmaxRho, const FORGET ALGO);

  void getMetrics(ostringstream& buff);
  void getHeaders(ostringstream& buff);
  void restart();

  void indexToSample(const int nSample, Uint& seq, Uint& obs) const;
  void sampleMultipleTrans(Uint* seq, Uint* obs, const Uint N, const int thrID);
  void sampleTransition(Uint& seq, Uint& obs, const int thrID);
  void sampleSequence(Uint& seq, const int thrID);
  void sampleTransitions_OPW(vector<Uint>& seq, vector<Uint>& obs);
  void sampleSequences(vector<Uint>& seq);

  inline Uint readNSeen() const
  {
    Uint ret;
    #pragma omp atomic read
    ret = nSeenTransitions;
    return ret;
    //#endif
  }
  inline Uint readNConcluded() const
  {
    Uint ret;
    #pragma omp atomic read
    ret = nCmplTransitions;
    return ret;
    //#endif
  }
  inline Uint readNSeenSeq() const
  {
    Uint ret;
    #pragma omp atomic read
    ret = nSeenSequences;
    return ret;
    //#endif
  }
  inline Uint readNData() const
  {
    Uint ret;
    #pragma omp atomic read
    ret = nTransitions;
    return ret;
  }
  inline Uint readNSeq() const
  {
    Uint ret;
    #pragma omp atomic read
    ret = nSequences;
    return ret;
  }

 private:
  inline void popBackSequence()
  {
    removeSequence( readNSeq() - 1 );
    Set.pop_back();
    #pragma omp atomic
    nSequences--;
    assert(nSequences==Set.size());
  }
  inline void pushBackSequence(Sequence*const seq)
  {
    lock_guard<mutex> lock(dataset_mutex);
    Set.push_back(nullptr);
    addSequence( readNSeq(), seq);
    #pragma omp atomic
    nSequences++;
    assert( readNSeq() == Set.size());
  }
  inline void addSequence(const Uint ind, Sequence*const seq)
  {
    assert(Set[ind] == nullptr && seq not_eq nullptr);
    #pragma omp atomic
    nTransitions += seq->ndata();
    Set[ind] = seq;
  }
  inline void removeSequence(const Uint ind)
  {
    assert(Set[ind] not_eq nullptr);
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
