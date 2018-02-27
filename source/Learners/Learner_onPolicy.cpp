/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner_onPolicy.h"

Learner_onPolicy::Learner_onPolicy(Environment*const _env, Settings&_s): Learner(_env, _s), nHorizon(_s.maxTotObsNum),
nEpochs(_s.batchSize/_s.obsPerStep) {
  cout << "nHorizon:"<<nHorizon<<endl;
  cout << "nEpochs:"<<nEpochs<<endl;
}
// nHorizon is number of obs in buffer during training
// steps per epoch = nEpochs * nHorizon / batchSize
// obs per step = nHorizon / (steps per epoch)
// this leads to formula used to compute nEpochs

void Learner_onPolicy::prepareData()
{
  if(cntEpoch >= nEpochs) {
    data->clearAll();
    //reset batch learning counters
    cntEpoch = 0; cntBatch = 0;
    updateComplete = false;
    updatePrepared = false;
  }
}

// unlockQueue tells scheduler that has stopped receiving states from slaves
// whether should start communication again.
// for on policy learning, when enough data is collected from slaves
// gradient stepping starts and collection is paused
// when training is concluded collection restarts
bool Learner_onPolicy::lockQueue() const
{
  return bTrain && data->readNData() >= nHorizon;
}

void Learner_onPolicy::spawnTrainTasks_seq()
{
  if( updateComplete ) die("undefined behavior");
  if( data->readNData() < nHorizon ) die("undefined behavior");
  if( not bTrain ) return;

  updatePrepared = true;

  #pragma omp parallel for schedule(dynamic)
  for (Uint i=0; i<batchSize; i++)
  {
    #pragma omp task
    {
      Uint seq, obs;
      const int thrID = omp_get_thread_num();

      if(thrID==0) profiler_ext->stop_start("WORK");

      data->sampleTransition(seq, obs, thrID);
      Train(seq, obs, thrID);

      if(thrID==0) profiler_ext->stop_start("COMM");

      input->gradient(thrID);
    }
  }
  updateComplete = true;
}

void Learner_onPolicy::spawnTrainTasks_par() { }

void Learner_onPolicy::prepareGradient()
{
  if (updateComplete && bTrain) {
    cntBatch += batchSize;
    if(cntBatch >= nHorizon) {
      cntBatch = 0;
      cntEpoch++;
    }
  }
  Learner::prepareGradient();
}
