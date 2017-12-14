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
    if(bSampleSequences) return 8*batchSize;
    else                 return   batchSize;
  }
  inline Uint read_nData() const
  {
    const Uint _nData = data->readNSeen();
    if(_nData < nData_b4PolUpdates) return 0;
    return _nData - nData_b4PolUpdates;
  }

  inline void resample(const Uint thrID) const // TODO resample sequence
  {
    int newSample = -1;
    #pragma omp critical
    newSample = data->sample(thrID);

    if(newSample >= 0) // process the other sample
    {
      Uint sequence, transition;
      data->indexToSample(newSample, sequence, transition);
      return Train(sequence, transition, thrID);
    }
    else return; // skip element of the batch -> as if added 0 gradient
  }

  //main training functions:
  void prepareData() override;
  bool unlockQueue() override;
  int spawnTrainTasks() override;
  void prepareGradient() override;
  bool batchGradientReady() override;
};
