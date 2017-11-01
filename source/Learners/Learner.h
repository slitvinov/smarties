/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "../Agent.h"
#include "../Environments/Environment.h"
#include "Transitions.h"
#include "../Network/Builder.h"
#include "../Network/Network.h"
#include "../Network/Optimizer.h"
#include <list>

using namespace std;

struct trainData
{
  trainData() : weight(1), MSE(0), avgQ(0), stdQ(0), minQ(1e9), maxQ(-1e9), relE(0), dCnt(0), epochCount(0) {}
  long double weight, MSE, avgQ, stdQ, minQ, maxQ, relE, dCnt;
  Uint epochCount;
};

class Learner
{
protected:
  const MPI_Comm mastersComm;
  Environment * const env;
  const Uint tgtUpdateDelay, nAgents, batchSize, nAppended, maxTotSeqNum, totNumSteps, nThreads, nSlaves;
  const int nSThreads, learn_rank, learn_size;
  Uint nInputs, nOutputs;
  const bool bRecurrent, bSampleSequences, bTrain;
  const Real tgtUpdateAlpha, greedyEps, gamma, epsAnneal, obsPerStep_orig;

  vector<Uint> sequences, transitions;

  Uint policyVecDim = 0;
  Uint cntUpdateDelay = 0, taskCounter = batchSize, nAddedGradients = 0;
  unsigned long nData_last = 0, nStep_last = 0;
  Real obsPerStep = obsPerStep_orig;
  Uint nData_b4PolUpdates = 0;

  ActionInfo aInfo;
  StateInfo  sInfo;
  mt19937* const gen;  //only ok if only thread 0 accesses
  Network* net;
  Optimizer* opt;
  Transitions* data;

public:
  Profiler* profiler;
  Profiler* profiler_ext = nullptr;
  mutable std::mutex task_mutex;
  int nTasks = 0;

protected:
  virtual void Train_BPTT(const Uint seq, const Uint thrID) const = 0;
  virtual void Train(const Uint seq, const Uint samp, const Uint thrID)const=0;
  virtual void stackAndUpdateNNWeights() = 0;
  virtual void updateTargetNetwork() = 0;
  virtual void processStats() = 0;

  Uint sampleSequences(vector<Uint>& sequences);
  Uint sampleTransitions(vector<Uint>& sequences, vector<Uint>& transitions);

public:
  Learner(MPI_Comm mastersComm, Environment*const env, Settings & settings);

  virtual ~Learner()
  {
    _dispose_object(profiler);
    _dispose_object(net);
    _dispose_object(opt);
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

  inline unsigned nData() const
  {
    return data->readNTransitions();
  }
  inline unsigned iter() const
  {
    return opt->nepoch;
  }
  inline bool reachedMaxGradStep() const
  {
    return opt->nepoch > totNumSteps;
  }

  inline Real annealingFactor() const
  {
    //number that goes from 1 to 0 with optimizer's steps
    assert(epsAnneal>1.);
    if(opt->nepoch >= epsAnneal || !bTrain) return 0;
    else return 1 - opt->nepoch/epsAnneal;
  }
  //*
  inline Real annealedGamma() const
  {
    //assert(epsAnneal>1. && bTrain && gamma>0.5);
    //if (opt->nepoch > epsAnneal) return gamma;
    //const Real anneal = 0.5 + 0.5*opt->nepoch/epsAnneal;
    //return anneal*gamma;
    return gamma;
  }
  /*/
  inline Real annealedGamma() const
  {
    assert(epsAnneal>1. && bTrain);
    if (opt->nepoch > epsAnneal) return gamma;
    const Real anneal = opt->nepoch/epsAnneal;
    const Real fac = 10 + anneal*(1./(1-gamma) -10);
    return 1 - 1./fac;
  }
  //*/

  inline bool readyForTrain() const
  {
    if(bSampleSequences)
    {
      if(data->adapt_TotSeqNum <= batchSize)
        die("I do not have enough data for training. Change hyperparameters");

      return bTrain && data->nSequences >= nSequences4Train();
    }
    else
    {
      if(data->adapt_TotSeqNum <= batchSize)
        die("I do not have enough data for training. Change hyperparameters");
     //const Uint nTransitions = data->readNTransitions();
     //if(data->nSequences>=data->adapt_TotSeqNum && nTransitions<nData_b4Train())
     //  die("I do not have enough data for training. Change hyperparameters");
     //const Real nReq = std::sqrt(data->readAvgSeqLen()*16)*batchSize;
     return bTrain && data->nSequences >= nSequences4Train();
    }
  }

  inline Uint nSequences4Train() const
  {
    return batchSize/learn_size;
  }

  inline Uint read_nData() const
  {
    const Uint _nData = data->readNSeen();
    if(_nData < nData_b4PolUpdates) return 0;
    return _nData - nData_b4PolUpdates;
  }

  virtual void select(const int agentId, const Agent& agent) = 0;
  virtual void dumpPolicy() = 0;

  //main training functions:
  virtual int spawnTrainTasks(const int availTasks);
  virtual void applyGradient();
  virtual void prepareData();

  //mass-handing of unfinished sequences from master
  void clearFailedSim(const int agentOne, const int agentEnd);
  void pushBackEndedSim(const int agentOne, const int agentEnd);

  //checks on status:
  virtual bool unlockQueue();
  virtual bool batchGradientReady();
  virtual bool readyForAgent(const int slave);
  virtual bool slaveHasUnfinishedSeqs(const int slave) const;

  void save(string name);
  void restart(string fname);

  void buildNetwork(Network*& _net , Optimizer*& _opt, const vector<Uint> nouts,
      Settings & settings, const vector<Uint> addedInputs = vector<Uint>(0));
};
