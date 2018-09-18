//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "MemoryBuffer.h"
#include "Approximator.h"

#include <list>
using namespace std;

class Learner
{
protected:
  Settings & settings;
  Environment * const env;
  const MPI_Comm mastersComm = settings.mastersComm;

  const bool bSampleSequences=settings.bSampleSequences, bTrain=settings.bTrain;
  const Uint totNumSteps=settings.totNumSteps, batchSize=settings.batchSize;
  const Uint policyVecDim=settings.policyVecDim, nAgents=settings.nAgents;
  const Real learnR=settings.learnrate, gamma=settings.gamma;
  const Uint nThreads=settings.nThreads, nWorkers=settings.nWorkers;
  const Real CmaxPol=settings.clipImpWeight, ReFtol=settings.penalTol;
  const Real explNoise=settings.explNoise, epsAnneal=settings.epsAnneal;
  const int learn_rank=settings.learner_rank, learn_size=settings.learner_size;
  std::atomic<long int> _nStep{0};

  std::atomic<long int> nData_b4Startup{0};
  std::atomic<long int> nStep_b4Startup{0};
  std::atomic<Uint> nAddedGradients{0};

  mutable bool updateComplete = false;
  mutable bool updateToApply = false;

  const ActionInfo& aInfo = env->aI;
  const StateInfo&  sInfo = env->sI;
  std::vector<std::mt19937>& generators = settings.generators;
  MemoryBuffer* data;
  Encapsulator* input;
  TrainData* trainInfo = nullptr;
  vector<Approximator*> F;
  mutable std::mutex buffer_mutex;

  virtual void processStats();

public:
  Profiler* profiler = nullptr;
  string learner_name;
  Uint learnID;

  Learner(Environment*const env, Settings & settings);

  virtual ~Learner() {
    _dispose_object(profiler);
    _dispose_object(data);
  }

  inline void setLearnerName(const string lName, const Uint id) {
    learner_name = lName;
    data->learnID = id;
    learnID = id;
  }

  inline unsigned tStepsTrain() const {
    return data->readNSeen() - nData_b4Startup;
  }
  inline unsigned nSeqsEval() const {
    return data->readNSeenSeq();
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

  virtual bool lockQueue() const = 0;

  virtual void prepareGradient();
  virtual void applyGradient();
  virtual void initializeLearner();
  bool predefinedNetwork(Builder& input_net, const Uint privateNum = 1);
  virtual void save();
  virtual void restart();
};
