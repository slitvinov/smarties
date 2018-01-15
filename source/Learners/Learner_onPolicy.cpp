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
nEpochs(_s.obsPerStep * _s.batchSize) { }
// nHorizon is number of obs in buffer during training
// steps per epoch = nEpochs * nHorizon / batchSize
// obs per step = nHorizon / (steps per epoch)
// this leads to formula used to compute nEpochs

void Learner_onPolicy::prepareData()
{
  if(cntEpoch == nEpochs) {
    data->clearAll();
    //reset batch learning counters
    cntTrajectories = 0; cntHorizon = 0; cntEpoch = 0; cntBatch = 0;
    waitingForData = true;
    updateComplete = false;
    updatePrepared = false;
  }
}

bool Learner_onPolicy::batchGradientReady()
{
  updateComplete = taskCounter >= batchSize;
  return updateComplete;
}

// unlockQueue tells scheduler that has stopped receiving states from slaves
// whether should start communication again.
// for on policy learning, when enough data is collected from slaves
// gradient stepping starts and collection is paused
// when training is concluded collection restarts
bool Learner_onPolicy::unlockQueue()
{
  return waitingForData;
}

int Learner_onPolicy::spawnTrainTasks()
{
  if( updateComplete || cntHorizon < nHorizon  || not bTrain )
    return 0;
  waitingForData = false;
  updateComplete = false;
  updatePrepared = true;

  #pragma omp parallel for schedule(dynamic)
  for (Uint i=0; i<batchSize; i++)
  {
    addToNTasks(1);
    #pragma omp task
    {
      Uint seq, obs;
      const int thrID = omp_get_thread_num();
      if(thrID==0) profiler_ext->stop_start("WORK");
      data->sampleTransition(seq, obs, thrID);
      Train(seq, obs, thrID);
      if(thrID==0) profiler_ext->stop_start("COMM");
      #pragma omp atomic
      taskCounter++;
      //printf("Thread %d done %u %u\n",thrID,seq,obs); fflush(0);
      if(taskCounter >= batchSize) updateComplete = true;
      addToNTasks(-1);
    }
  }
  return 0;
}

void Learner_onPolicy::prepareGradient()
{
  if (updateComplete && bTrain) {
    taskCounter = 0;
    cntBatch += batchSize*learn_size;
    if(cntBatch >= nHorizon) {
      cntBatch = 0;
      cntEpoch++;
    }
  }
  Learner::prepareGradient();
}
