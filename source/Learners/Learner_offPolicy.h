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
  mutable Uint nSkipped = 0;
  Real nData_last = 0, nStep_last = 0;
  Real obsPerStep = obsPerStep_orig;
  vector<Uint> samp_seq, samp_obs;

public:
  Learner_offPolicy(Environment*const env, Settings& _s);

  bool readyForTrain() const;

  inline void resample(const Uint thrID) const // TODO resample sequence
  {
    Uint _nSkipped;
    #pragma omp atomic read
      _nSkipped = nSkipped;

    // If skipping too many samples return w/o sample to avoid code hanging.
    // If true smth is wrong. Approximator will print to screen a warning.
    if(_nSkipped >= batchSize) return;

    #pragma omp atomic
    nSkipped++;

    Uint sequence, transition;
    data->sampleTransition(sequence, transition, thrID);
    data->Set[sequence]->setSampled(transition);
    return Train(sequence, transition, thrID);
  }

  //main training functions:
  void prepareData() override;
  bool lockQueue() const override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  void prepareGradient() override;
};
