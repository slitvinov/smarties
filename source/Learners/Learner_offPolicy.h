/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Learner.h"

class Learner_offPolicy: public Learner
{
protected:
  const Real obsPerStep_orig;
  const Uint nObsPerTraining;
  Uint taskCounter = 0, nData_b4Startup = 0;
  Real nData_last = 0, nStep_last = 0;
  Real obsPerStep = obsPerStep_orig;
  vector<Uint> samp_seq, samp_obs;

public:
  Learner_offPolicy(Environment*const env, Settings& _s);

  bool readyForTrain() const;

  inline void advanceCounters() {
    //shift data / gradient counters to maintain grad stepping to sample
    // collection ratio prescirbed by obsPerStep
    const Real stepCounter = nStep - (Real)nStep_last;
    nData_last += stepCounter*obsPerStep;
    nStep_last = nStep;
  }

  //main training functions:
  void prepareData() override;
  bool lockQueue() const override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  void prepareGradient() override;
};
