//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "../ReplayMemory/MemoryBuffer.h"
#include "../ReplayMemory/Collector.h"
#include "../ReplayMemory/MemoryProcessing.h"
#include "../Utils/Profiler.h"


class Learner
{
 protected:
  const Uint freqPrint = 1000;
  Settings & settings;
  Environment * const env;

 public:
  const MPI_Comm mastersComm = settings.mastersComm;

  const Uint nObsB4StartTraining = settings.minTotObsNum_loc;
  const bool bTrain = settings.bTrain;
  const Real obsPerStep_loc = settings.obsPerStep_loc;

  const Uint policyVecDim = env->aI.policyVecDim;
  const Uint nAgents = settings.nAgents;
  const Uint nThreads = settings.nThreads;

  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;
  const int dropRule = settings.nnPdrop;
  // hyper-parameters:
  const Uint totNumSteps = settings.totNumSteps;

  const Real gamma = settings.gamma;
  const Real CmaxPol = settings.clipImpWeight;
  const Real ReFtol = settings.penalTol;
  const Real epsAnneal = settings.epsAnneal;

  const FORGET ERFILTER =
    MemoryProcessing::readERfilterAlgo(settings.ERoldSeqFilter, CmaxPol>0);
  DelayedReductor<long double> ReFER_reduce = DelayedReductor<long double>(settings, LDvec{ 0.0, 1.0 } );

  const StateInfo&  sInfo = env->sI;
  const ActionInfo& aInfo = env->aI;
  const ActionInfo* const aI = &aInfo;

 protected:
  long nDataGatheredB4Startup = 0;
  int algoSubStepID = -1;

  Real alpha = 0.5; // weight between critic and policy
  Real beta = CmaxPol<=0? 1 : 0.0; // if CmaxPol==0 do naive Exp Replay
  Real CmaxRet = 1 + CmaxPol;
  Real CinvRet = 1 / CmaxRet;
  bool computeQretrace = false;

  std::atomic<long> _nGradSteps{0};

  std::vector<std::mt19937>& generators = settings.generators;

  MemoryBuffer* const data = new MemoryBuffer(settings, env);
  MemoryProcessing* const data_proc = new MemoryProcessing(settings, data);
  Collector* const data_get = new Collector(settings, data);
  Profiler* const profiler = new Profiler();

  TrainData* trainInfo = nullptr;
  mutable std::mutex buffer_mutex;

  virtual void processStats();

 public:
  std::string learner_name;
  Uint learnID;

  Learner(Environment*const env, Settings & settings);

  virtual ~Learner() {
    _dispose_object(data_proc);
    _dispose_object(data_get);
    _dispose_object(data);
  }

  inline void setLearnerName(const std::string lName, const Uint id) {
    learner_name = lName;
    data->learnID = id;
    learnID = id;
  }

  inline long nLocTimeStepsTrain() const {
    return data->readNSeen_loc() - nDataGatheredB4Startup;
  }
  inline long locDataSetSize() const {
    return data->readNData();
  }
  inline unsigned nSeqsEval() const {
    return data->readNSeenSeq_loc();
  }
  inline long int nGradSteps() const {
    return _nGradSteps.load();
  }
  inline Real annealingFactor() const {
    //number that goes from 1 to 0 with optimizer's steps
    assert(epsAnneal>1.);
    const auto mynstep = nGradSteps();
    if(mynstep*epsAnneal >= 1 || !bTrain) return 0;
    else return 1 - mynstep*epsAnneal;
  }

  virtual void select(Agent& agent) = 0;
  virtual void setupTasks(TaskQueue& tasks) = 0;

  virtual void globalGradCounterUpdate();

  virtual bool blockDataAcquisition() const;
  virtual bool unblockGradientUpdates() const;

  void processMemoryBuffer();
  void updateRetraceEstimates();
  void finalizeMemoryProcessing();
  virtual void initializeLearner();

  virtual void logStats();

  virtual void getMetrics(std::ostringstream& buff) const;
  virtual void getHeaders(std::ostringstream& buff) const;

  virtual void save();
  virtual void restart();

 protected:
  inline void backPropRetrace(Sequence*const S, const Uint t) {
    if(t == 0) return;
    const Fval W = S->offPolicImpW[t], R=data->scaledReward(S, t), G = gamma;
    const Fval C = W<1 ? W:1, V = S->state_vals[t], A = S->action_adv[t];
    S->setRetrace(t-1, R + G*V + G*C*(S->Q_RET[t] -A-V) );
  }
  inline Fval updateRetrace(Sequence*const S, const Uint t, const Fval A,
    const Fval V, const Fval W) const {
    assert(W >= 0);
    if(t == 0) return 0;
    S->setStateValue(t, V); S->setAdvantage(t, A);
    const Fval oldRet = S->Q_RET[t-1], C = W<1 ? W:1, G = gamma;
    S->setRetrace(t-1, data->scaledReward(S,t) +G*V + G*C*(S->Q_RET[t] -A-V) );
    return std::fabs(S->Q_RET[t-1] - oldRet);
  }
};
