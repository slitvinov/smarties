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

  inline bool readyForTrain() const
  {
    if(bSampleSequences) {
      if(data->adapt_TotSeqNum <= batchSize/learn_size)
        die("I do not have enough data for training. Change hyperparameters");

      return bTrain && data->nSequences >= nSequences4Train();
    } else {
      if(data->adapt_TotSeqNum <= batchSize/learn_size)
        die("I do not have enough data for training. Change hyperparameters");
     //const Uint nTransitions = data->readNTransitions();
     //if(data->nSequences>=data->adapt_TotSeqNum && nTransitions<nData_b4Train())
     //  die("I do not have enough data for training. Change hyperparameters");
     //const Real nReq = std::sqrt(data->readAvgSeqLen()*16)*batchSize;
     return bTrain && data->nSequences >= nSequences4Train();
    }
  }
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
