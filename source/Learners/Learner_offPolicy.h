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
  const Real obsPerStep_loc = settings.obsPerStep_loc;

  Real alpha = 0.5; // weight between critic and policy
  Real beta = CmaxPol<=0? 1 : 0.0; // if CmaxPol==0 do naive Exp Replay
  Real CmaxRet = 1 + CmaxPol;
  Real CinvRet = 1 / CmaxRet;

  const FORGET ERFILTER =
    MemoryProcessing::readERfilterAlgo(settings.ERoldSeqFilter, CmaxPol>0);

  DelayedReductor ReFER_reduce = DelayedReductor(settings, LDvec{ 0.0, 1.0 } );
public:
  Learner_offPolicy(Environment*const env, Settings& _s);
  virtual void TrainBySequences(const Uint seq, const Uint wID,
    const Uint bID, const Uint tID) const = 0;
  virtual void Train(const Uint seq, const Uint samp, const Uint wID,
    const Uint bID, const Uint tID) const = 0;

  //main training functions:
  bool blockDataAcquisition() const override;
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
