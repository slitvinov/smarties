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

class Learner_onPolicy: public Learner
{
protected:
  const Uint nHorizon, nEpochs;
  Uint cntHorizon = 0, cntTrajectories = 0, cntEpoch = 0;
  mutable Uint nSkipped = 0, cntBatch = 0;

public:
  Learner_onPolicy(Environment*const _env, Settings&_s);

  //main training functions:
  void prepareData() override;
  bool unlockQueue() override;
  int spawnTrainTasks() override;
  void prepareGradient() override;
  bool batchGradientReady() override;
  inline void resample(const Uint thrID) const // TODO resample sequence
  {
    // skipping too many samples, something is wrong. To avoid code hanging return: 
    if(nSkipped>=batchSize) return;

    #pragma omp atomic
    nSkipped++; 

    Uint sequence, transition;
    data->sampleTransition(sequence, transition, thrID);
    data->Set[sequence]->setSampled(transition);
    return Train(sequence, transition, thrID);
  }
};
