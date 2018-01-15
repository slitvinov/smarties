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
  const Uint nObsPerTraining;
  const Real obsPerStep_orig;
  Uint taskCounter = batchSize, nData_b4PolUpdates = 0, nToSpawn = 0;
  unsigned long nData_last = 0, nStep_last = 0;
  Real obsPerStep = obsPerStep_orig, nStoredSeqs_last = 0;

public:
  Learner_offPolicy(Environment*const env, Settings& _s);

  bool readyForTrain() const;

  inline Uint nSequences4Train() const
  {
    return nObsPerTraining;
    //if(bSampleSequences) return 8*batchSize;
    //else                 return   batchSize/2;
  }
  inline Uint read_nData() const
  {
    const Uint _nData = data->readNSeen();
    if(_nData < nData_b4PolUpdates) return 0;
    return _nData - nData_b4PolUpdates;
  }

  inline void resample(const Uint thrID) const // TODO resample sequence
  {
    Uint sequence, transition;
    data->sampleTransition(sequence, transition, thrID);
    return Train(sequence, transition, thrID);
  }

  //main training functions:
  void prepareData() override;
  bool unlockQueue() override;
  int spawnTrainTasks() override;
  void prepareGradient() override;
  bool batchGradientReady() override;
};
