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

#include <list>

class Learner
{
 protected:
  Settings & settings;
  Environment * const env;

  long nData_b4Startup = 0;
  std::atomic<long> _nStep{0}, _nData{0};
  std::atomic<bool> updatedNdata{false};
  std::atomic<Uint> nAddedGradients{0};

  bool updateComplete = false;
  bool updateToApply = false;

  std::vector<std::mt19937>& generators = settings.generators;

  MemoryBuffer* const data = new MemoryBuffer(settings, env);
  Collector* const data_get = new Collector(settings, this, data);
  MemoryProcessing* const data_proc = new MemoryProcessing(settings, data);
  Encapsulator * const input = new Encapsulator("input", settings, data);

  TrainData* trainInfo = nullptr;
  std::vector<Approximator*> F;
  mutable std::mutex buffer_mutex;

  virtual void processStats();

 public:
  const MPI_Comm mastersComm = settings.mastersComm;

  const bool bSampleSequences = settings.bSampleSequences;
  const bool bTrain = settings.bTrain;

  const Uint policyVecDim = settings.policyVecDim;
  const Uint nAgents = settings.nAgents;
  const Uint nThreads = settings.nThreads;

  const int learn_rank = settings.learner_rank;
  const int learn_size = settings.learner_size;

  // hyper-parameters:
  const Uint batchSize = settings.batchSize;
  const Uint totNumSteps = settings.totNumSteps;

  const Real learnR = settings.learnrate;
  const Real gamma = settings.gamma;
  const Real CmaxPol = settings.clipImpWeight;
  const Real ReFtol = settings.penalTol;
  const Real explNoise = settings.explNoise;
  const Real epsAnneal = settings.epsAnneal;

  const ActionInfo& aInfo = env->aI;
  const StateInfo&  sInfo = env->sI;

  Profiler* profiler = nullptr;
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

  inline unsigned tStepsTrain() const {
    return data->readNSeen_loc() - nData_b4Startup;
  }
  inline unsigned nSeqsEval() const {
    return data->readNSeenSeq_loc();
  }
  inline unsigned nData() const {
    return data->readNData();
  }
  inline long int nStep() const {
    return _nStep.load();
  }
  inline Real annealingFactor() const {
    //number that goes from 1 to 0 with optimizer's steps
    assert(epsAnneal>1.);
    const auto mynstep = nStep();
    if(mynstep*epsAnneal >= 1 || !bTrain) return 0;
    else return 1 - mynstep*epsAnneal;
  }

  virtual void select(Agent& agent) = 0;
  virtual void TrainBySequences(const Uint seq, const Uint wID,
    const Uint bID, const Uint tID) const = 0;
  virtual void Train(const Uint seq, const Uint samp, const Uint wID,
    const Uint bID, const Uint tID) const = 0;

  virtual void getMetrics(ostringstream& buff) const;
  virtual void getHeaders(ostringstream& buff) const;

  //main training loop functions:
  virtual void spawnTrainTasks_par() = 0;
  virtual void spawnTrainTasks_seq() = 0;
  virtual bool bNeedSequentialTrain() = 0;

  void globalDataCounterUpdate(const long globalSeenTransitions) {
    _nData = globalSeenTransitions;
    updatedNdata = true;
  }
  void globalGradCounterUpdate() {
    _nStep++;
    updatedNdata = false;
  }

  bool unblockGradStep() const {
    if(updatedNdata not_eq true) return false;
    //assert( _nData.load() - nData_b4Startup > _nStep.load() * obsPerStep );
    assert( blockDataAcquisition() );
    return true;
  }
  virtual bool blockDataAcquisition() const = 0;

  virtual void prepareGradient();
  virtual void applyGradient();
  virtual void initializeLearner();
  bool predefinedNetwork(Builder& input_net, const Uint privateNum = 1);
  virtual void save();
  virtual void restart();
};
