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
  mutable Uint cntBatch = 0, cntEpoch = 0;

public:
  Learner_onPolicy(Environment*const _env, Settings&_s);

  //main training functions:
  void prepareData() override;
  bool lockQueue() const override;
  void spawnTrainTasks_seq() override;
  void spawnTrainTasks_par() override;
  void prepareGradient() override;
};
