//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Learner_h
#define smarties_Learner_h

#include "Core/StateAction.h"
#include "Utils/Profiler.h"
#include "ReplayMemory/MemoryBuffer.h"
#include "Utils/StatsTracker.h"
#include "Utils/TaskQueue.h"
#include "Settings.h"

namespace smarties
{

class MemoryProcessing;
class Collector;

class Learner
{
protected:
  const Uint freqPrint = 1000;
  DistributionInfo & distrib;
  Settings & settings;
  MDPdescriptor & MDP;

public:
  const MPI_Comm learnersComm = distrib.learners_train_comm;
  const Uint learn_rank = MPICommRank(learnersComm);
  const Uint learn_size = MPICommSize(learnersComm);
  const Uint nThreads = distrib.nThreads;
  const Uint nAgents = distrib.nAgents;

  const Uint policyVecDim = MDP.policyVecDim;
  const ActionInfo aInfo = ActionInfo(MDP);
  const StateInfo  sInfo = StateInfo(MDP);

  // training loop scheduling:
  const Uint totNumSteps = settings.totNumSteps;
  const Real obsPerStep_loc = settings.obsPerStep_local;
  const long nObsB4StartTraining = settings.minTotObsNum_local;
  const bool bTrain = settings.bTrain;

  // some algorithm hyper-parameters:
  const Real gamma = settings.gamma;
  const Real ReFtol = settings.penalTol;
  const Real CmaxPol = settings.clipImpWeight;
  const Real epsAnneal = settings.epsAnneal;

  const FORGET ERFILTER;
  DelayedReductor<long double> ReFER_reduce =
                      DelayedReductor<long double>(distrib, LDvec{ 0.0, 1.0 } );


protected:
  long nDataGatheredB4Startup = 0;
  int algoSubStepID = -1;

  Real alpha = 0.5; // weight between critic and policy
  Real beta = CmaxPol<=0? 1 : 0.0; // if CmaxPol==0 do naive Exp Replay
  Real CmaxRet = 1 + CmaxPol;
  Real CinvRet = 1 / CmaxRet;
  bool computeQretrace = false;

  std::atomic<long> _nGradSteps{0};

  std::vector<std::mt19937>& generators = distrib.generators;

  const std::unique_ptr<MemoryBuffer> data =
                         std::make_unique<MemoryBuffer>(MDP, settings, distrib);
  MemoryProcessing* const data_proc;
  Collector* const        data_get;
  const std::unique_ptr<Profiler> profiler  = std::make_unique<Profiler>();

  TrainData* trainInfo = nullptr;
  mutable std::mutex buffer_mutex;

  virtual void processStats();

public:
  std::string learner_name;
  Uint learnID;

  Learner(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_);
  virtual ~Learner();

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
  inline long nGradSteps() const {
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
  void backPropRetrace(Sequence& S, const Uint t)
  {
    if(t == 0) return;
    const Fval W = S.offPolicImpW[t];
    const Fval R = data->scaledReward(S, t), G = gamma;
    const Fval C = W<1 ? W:1, V = S.state_vals[t], A = S.action_adv[t];
    S.setRetrace(t-1, R + G*V + G*C*(S.Q_RET[t] -A-V) );
  }
  Fval updateRetrace(Sequence& S, const Uint t,
                     const Fval A, const Fval V, const Fval W) const
  {
    assert(W >= 0);
    if(t == 0) return 0;
    S.setStateValue(t, V); S.setAdvantage(t, A);
    const Fval oldRet = S.Q_RET[t-1], C = W<1 ? W:1, G = gamma;
    S.setRetrace(t-1, data->scaledReward(S,t) +G*V + G*C*(S.Q_RET[t] -A-V) );
    return std::fabs(S.Q_RET[t-1] - oldRet);
  }
};

}
#endif
