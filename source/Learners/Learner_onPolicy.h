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
  const Uint nEpochs = 10, nHorizon = 4096;
  Uint cntHorizon = 0, cntTrajectories = 0, cntEpoch = 0, cntBatch = 0;
  Uint taskCounter = 0;

public:
  Learner_onPolicy(Environment*const _env, Settings&_s);

  //main training functions:
  void prepareData() override;
  bool unlockQueue() override;
  int spawnTrainTasks() override;
  void prepareGradient() override;
  bool batchGradientReady() override;
};
