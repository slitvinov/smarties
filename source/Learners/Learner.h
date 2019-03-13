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
#include "../Network/Approximator.h"


class Learner
{
 protected:
  const Uint freqPrint = 1000;
  Settings & settings;
  Environment * const env;
  int algoSubStepID = -1;
 public:
  const MPI_Comm mastersComm = settings.mastersComm;

  const bool bSampleSequences = settings.bSampleSequences;
  const Uint nObsB4StartTraining = settings.minTotObsNum_loc;
  const bool bTrain = settings.bTrain;

  const Uint policyVecDim = env->aI.policyVecDim;
  const Uint nAgents = settings.nAgents;
  const Uint nThreads = settings.nThreads;

  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;
  const int dropRule = settings.nnPdrop;
  // hyper-parameters:
  const Uint batchSize = settings.batchSize_loc;
  const Uint totNumSteps = settings.totNumSteps;
  const Uint ESpopSize = settings.ESpopSize;

  const Real learnR = settings.learnrate;
  const Real gamma = settings.gamma;
  const Real CmaxPol = settings.clipImpWeight;
  const Real ReFtol = settings.penalTol;
  const Real explNoise = settings.explNoise;
  const Real epsAnneal = settings.epsAnneal;

  const StateInfo&  sInfo = env->sI;
  const ActionInfo& aInfo = env->aI;
  const ActionInfo* const aI = &aInfo;

 protected:
  long nDataGatheredB4Startup = 0;
  std::atomic<long> _nGradSteps{0};

  std::vector<std::mt19937>& generators = settings.generators;

  MemoryBuffer* const data = new MemoryBuffer(settings, env);
  Encapsulator * const input = new Encapsulator("input", settings, data);
  MemoryProcessing* const data_proc = new MemoryProcessing(settings, data);
  Collector* const data_get = new Collector(settings, data);
  Profiler* const profiler = new Profiler();

  TrainData* trainInfo = nullptr;
  std::vector<Approximator*> F;
  mutable std::mutex buffer_mutex;

  virtual void processStats();
  void createSharedEncoder(const Uint privateNum = 1);
  bool predefinedNetwork(Builder& input_net, const Uint privateNum = 1);

 public:
  std::string learner_name;
  Uint learnID;

  Learner(Environment*const env, Settings & settings);

  virtual ~Learner() {
    _dispose_object(data_proc);
    _dispose_object(data_get);
    _dispose_object(input);
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
  virtual void getMetrics(ostringstream& buff) const;
  virtual void getHeaders(ostringstream& buff) const;

  void globalDataCounterUpdate();

  virtual void globalGradCounterUpdate();

  virtual bool blockDataAcquisition() const = 0;

  virtual void prepareGradient();
  virtual void applyGradient();
  virtual void logStats();
  virtual void save();
  virtual void restart();
};
