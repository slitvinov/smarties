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
  const Real greedyEps, epsAnneal, gamma;
  const int learn_rank, learn_size;
  unsigned long nStep = 0;
  Uint nAddedGradients = 0;

  ActionInfo aInfo;
  StateInfo  sInfo;
  std::vector<std::mt19937>& generators;
  MemoryBuffer* data;
  Encapsulator* input;
  vector<Approximator*> F;

  trainData stats;
  mutable vector<trainData> Vstats;
  virtual void processStats();

public:
  Profiler* profiler;
  Profiler* profiler_ext = nullptr;
  mutable std::mutex task_mutex;
  int nTasks = 0;

  Learner(Environment*const env, Settings & settings);

  virtual ~Learner() {
    _dispose_object(profiler);
    _dispose_object(data);
  }

  inline int readNTasks() const
  {
    //lock_guard<mutex> lock(task_mutex);
    return nTasks;
  }
  inline void addToNTasks(const int add)
  {
    //lock_guard<mutex> lock(task_mutex);
    #pragma omp atomic
    nTasks += add;
  }
  inline unsigned iter() const
  {
    return nStep;
  }
  inline unsigned nData() const
  {
    return data->readNTransitions();
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

  virtual void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const;
  //mass-handing of unfinished sequences from master
  void clearFailedSim(const int agentOne, const int agentEnd);
  void pushBackEndedSim(const int agentOne, const int agentEnd);
  bool slaveHasUnfinishedSeqs(const int slave) const;

  //main training loop functions:
  virtual int spawnTrainTasks(const int availTasks) = 0;
  virtual void prepareData() = 0;
  virtual bool unlockQueue() = 0;
  virtual bool batchGradientReady() = 0;
  virtual bool readyForAgent(const int slave) = 0;

  virtual void prepareGradient();
  void synchronizeGradients();

  void restart();
};
