//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the “CC BY-SA 4.0” license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once

#include "Learner.h"

class Learner_offPolicy: public Learner
{
protected:
  const Real obsPerStep_orig;
  const Uint nObsPerTraining;
  Uint taskCounter = 0;
  mutable Uint percData = 0;
  Uint nData_b4Startup = 0;
  Real nData_last = 0, nStep_last = 0;
  Real obsPerStep = obsPerStep_orig;
  vector<Uint> samp_seq, samp_obs;


  Real beta = CmaxPol<=0? 1 : .2; // if CmaxPol==0 do naive Exp Replay
  Real CmaxRet = 1 + CmaxPol;

  ApproximateReductor<double, MPI_DOUBLE> reductor =
    ApproximateReductor<double, MPI_DOUBLE>(mastersComm, 2);
public:
  Learner_offPolicy(Environment*const env, Settings& _s);

  bool readyForTrain() const;
  bool stopGrads() const;

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
  void prepareGradientReFER();
};
