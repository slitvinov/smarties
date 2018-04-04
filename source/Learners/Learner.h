/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "MemoryBuffer.h"
#include "Approximator.h"

#include <list>
using namespace std;

class Learner
{
protected:
  const MPI_Comm mastersComm;
  Environment * const env;
  const bool bSampleSequences, bTrain;
  const Uint nAgents, batchSize, totNumSteps, nThreads, nSlaves, policyVecDim;
  const Real greedyEps, epsAnneal, gamma, CmaxPol;
  const int learn_rank, learn_size;
  unsigned long nStep = 0;
  Uint nAddedGradients = 0;
  mutable Uint nSkipped = 0;
  FORGET MEMBUF_FILTER_ALGO = OLDEST;

  mutable bool updatePrepared = false;
  mutable bool updateComplete = false;

  const ActionInfo aInfo;
  const StateInfo  sInfo;
  std::vector<std::mt19937>& generators;
  MemoryBuffer* data;
  Encapsulator* input;
  vector<Approximator*> F;
  mutable std::mutex buffer_mutex;

  trainData stats;
  mutable vector<trainData> Vstats;
  virtual void processStats();

  inline bool canSkip() const
  {
    Uint _nSkipped;
    #pragma omp atomic read
      _nSkipped = nSkipped;
    // If skipping too many samples return w/o sample to avoid code hanging.
    // If true smth is wrong. Approximator will print to screen a warning.
    return _nSkipped < batchSize;
  }

  inline void resample(const Uint thrID) const // TODO resample sequence
  {
    if( not canSkip() ) return;

    #pragma omp atomic
    nSkipped++;

    Uint sequence, transition;
    data->sampleTransition(sequence, transition, thrID);
    data->Set[sequence]->setSampled(transition);
    return Train(sequence, transition, thrID);
  }

public:
  Profiler* profiler;
  Profiler* profiler_ext = nullptr;
  int& nTasks;
  string learner_name;

  Learner(Environment*const env, Settings & settings);

  virtual ~Learner() {
    _dispose_object(profiler);
    _dispose_object(data);
  }

  inline void setLearnerName(const string lName) {
    learner_name = lName;
  }

  inline unsigned time() const
  {
    return data->readNSeen();
  }
  inline unsigned iter() const
  {
    return nStep;
  }
  inline unsigned nData() const
  {
    return data->readNData();
  }
  inline bool reachedMaxGradStep() const
  {
    return nStep > totNumSteps;
  }
  inline Real annealingFactor() const
  {
    //number that goes from 1 to 0 with optimizer's steps
    assert(epsAnneal>1.);
    if(nStep >= epsAnneal || !bTrain) return 0;
    else return 1 - nStep/epsAnneal;
  }

  virtual void select(const Agent& agent) = 0;
  virtual void Train_BPTT(const Uint seq, const Uint thrID) const = 0;
  virtual void Train(const Uint seq, const Uint samp, const Uint thrID) const=0;

  virtual void getMetrics(ostringstream& buff) const;
  virtual void getHeaders(ostringstream& buff) const;
  //mass-handing of unfinished sequences from master
  void clearFailedSim(const int agentOne, const int agentEnd);
  void pushBackEndedSim(const int agentOne, const int agentEnd);
  bool slaveHasUnfinishedSeqs(const int slave) const;

  //main training loop functions:
  virtual void spawnTrainTasks_par() = 0;
  virtual void spawnTrainTasks_seq() = 0;

  virtual void prepareData() = 0;
  virtual bool lockQueue() const = 0;

  virtual void prepareGradient();
  void synchronizeGradients();
  bool predefinedNetwork(Builder& input_net, Settings& settings);
  void restart();
};
