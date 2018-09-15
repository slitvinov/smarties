//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Learner.h"

class Learner_offPolicy: public Learner
{
protected:
  const Real obsPerStep_orig = settings.obsPerStep;
  const Uint nObsPerTraining = settings.minTotObsNum > settings.batchSize ?
                                  settings.minTotObsNum : settings.maxTotObsNum;
  const Uint ESpopSize = settings.ESpopSize;
  mutable int percData = -5;
  Real obsPerStep = obsPerStep_orig;

  Real alpha = 0.0; // weight between critic and policy
  Real beta = CmaxPol<=0? 1 : 0.0; // if CmaxPol==0 do naive Exp Replay
  Real CmaxRet = 1 + CmaxPol;
  Real CinvRet = 1 / CmaxRet;

  const FORGET ERFILTER =
    MemoryBuffer::readERfilterAlgo(settings.ERoldSeqFilter, CmaxPol>0);
  ApproximateReductor reductor = ApproximateReductor(mastersComm, 2);
public:
  Learner_offPolicy(Environment*const env, Settings& _s);

  bool readyForTrain() const;

  //main training functions:
  bool lockQueue() const override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  virtual void applyGradient() override;
  virtual void prepareGradient() override;
  bool bNeedSequentialTrain() override;
  virtual void initializeLearner() override;
  void save() override;
  void restart() override;

 protected:
  virtual void getMetrics(ostringstream& buff) const override;
  virtual void getHeaders(ostringstream& buff) const override;
};
