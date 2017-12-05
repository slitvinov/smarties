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
  vector<Uint> sequences, transitions;
  Uint taskCounter = batchSize, nData_b4PolUpdates = 0;
  unsigned long nData_last = 0, nStep_last = 0;
  Real obsPerStep = obsPerStep_orig, nStoredSeqs_last = 0;

public:
  Learner_offPolicy(Environment*const env, Settings& _s);

  bool readyForTrain() const;
  inline Uint nSequences4Train() const
  {
    return batchSize/learn_size;
  }
  inline Uint read_nData() const
  {
    const Uint _nData = data->readNSeen();
    if(_nData < nData_b4PolUpdates) return 0;
    return _nData - nData_b4PolUpdates;
  }

  //main training functions:
  int spawnTrainTasks(const int availTasks) override;
  void prepareData() override;
  bool unlockQueue() override;
  bool batchGradientReady() override;
  bool readyForAgent(const int slave) override;

  void applyGradient() override;
};
