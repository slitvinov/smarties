/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner_offPolicy.h"

Learner_offPolicy::Learner_offPolicy(Environment*const _env, Settings & _s) :
Learner(_env,_s), obsPerStep_orig(_s.obsPerStep) { }

void Learner_offPolicy::prepareData()
{
  if(data->adapt_TotSeqNum <= batchSize)
    die("I do not have enough data for training. Change hyperparameters")

  // it should be impossible to get here before starting batch update was ready.
  if(updateComplete) die("undefined behavior");

  // then no need to prepare a new update
  if(updatePrepared) return;

  // When is an algorithm not looking for more data and not ready for training?
  // It means that something messed with the transitions, ie by cancelling part
  // of the dataset, and therefore i need to reset how counters advance.
  // Anything that messes with MemoryBuffer can only happen on prepareGradient
  if(not waitingForData && not readyForTrain()) {
    abort();
    nData_b4PolUpdates = data->readNSeen();
    nStoredSeqs_last = nSequences4Train(); //RACER
    nData_last = 0;
  }

  // Do not prepare an update:
  if(    waitingForData || not readyForTrain()) {
    waitingForData = true;
    return;
  }


  if(data->requestUpdateSamples()) {
  profiler->stop_start("PRE1");
    data->updateActiveBuffer(); //update sampling //syncDataStats
  profiler->stop_start("PRE2");
    data->shuffle_samples();
  }

  if(nStep%100==0) {
    profiler->stop_start("PRE3");
    data->updateRewardsStats();
  }

  taskCounter = 0;
  sequences.resize(batchSize);
  transitions.resize(batchSize);

    profiler->stop_start("PRE4");
  nAddedGradients = bSampleSequences ? data->sampleSequences(sequences) :
    data->sampleTransitions(sequences, transitions);

  updatePrepared = true;

  profiler->stop_start("SLP");
}

bool Learner_offPolicy::readyForTrain() const
{
   //const Uint nTransitions = data->readNTransitions();
   //if(data->nSequences>=data->adapt_TotSeqNum && nTransitions<nData_b4Train())
   //  die("I do not have enough data for training. Change hyperparameters");
   //const Real nReq = std::sqrt(data->readAvgSeqLen()*16)*batchSize;
   return bTrain && data->nTransitions >= 6e4;//data->nSequences >= nSequences4Train();
}

bool Learner_offPolicy::batchGradientReady()
{
  //if there is not enough data for training: go back to communicating
  if(not readyForTrain()) {
    assert(waitingForData && not updatePrepared);
    nData_b4PolUpdates = data->readNSeen();
    return false;
  }

  if(not updatePrepared) { //then was waiting for data without an update ready
    waitingForData = false;
    return true; //we need to exit data loop to prepare an update
  }

  const Real _nData = read_nData();
  const Real dataCounter = _nData - std::min((Real)nData_last, _nData);
  const Real stepCounter =  nStep - (Real)nStep_last;
  //If I have done too many gradient steps on the avail data, go back to comm
  waitingForData = stepCounter*obsPerStep/learn_size > dataCounter;
  if(waitingForData) return false; // stay in data loop.

  updateComplete = taskCounter >= batchSize;
  return updateComplete;
}

// unlockQueue tells scheduler that has stopped receiving states from slaves
// whether should start communication again.
// for off policy learning, there is a ratio between gradient steps
// and observed transitions to be kept (approximatively) constant
bool Learner_offPolicy::unlockQueue()
{
  if( waitingForData ) return true;
  const Real _nData = read_nData();
  const Real dataCounter = _nData - std::min((Real)nData_last, _nData);
  const Real stepCounter = nStep  - (Real)nStep_last;
  const Real cushionData = nSlaves;
  // cushion leads collection of a bit more data than strictly necessary
  return stepCounter*obsPerStep/learn_size +cushionData >= dataCounter;
}

int Learner_offPolicy::spawnTrainTasks()
{
  if( updateComplete || not updatePrepared ) return 0;

  for (Uint i=0; i<sequences.size(); i++)
  {
    const Uint seq = sequences.back(); sequences.pop_back();
    const Uint obs = transitions.back(); transitions.pop_back();
    addToNTasks(1);
    #pragma omp task firstprivate(seq, obs)
    {
      const int thrID = omp_get_thread_num();
      //printf("Thread %d doing %u %u\n",thrID,seq,obs); fflush(0);
      if(thrID == 0) profiler_ext->stop_start("WORK");

      if(bSampleSequences) Train_BPTT(seq, thrID);
      else                 Train(seq, obs, thrID);

      if(thrID == 0) profiler_ext->stop_start("COMM");
      addToNTasks(-1);
      #pragma omp atomic
      taskCounter++;
      //printf("Thread %d done %u %u\n",thrID,seq,obs); fflush(0);
      if(taskCounter >= batchSize) updateComplete = true;
    }
  }
  return 0;
}

void Learner_offPolicy::prepareGradient()
{
  if(updateComplete) {
    //shift data / gradient counters to maintain grad stepping to sample
    // collection ratio prescirbed by obsPerStep
    const Real stepCounter = nStep+1 - (Real)nStep_last;
    nData_last += stepCounter*obsPerStep/learn_size;
    nStep_last = nStep+1; //actual counter advanced by base class
  }

  Learner::prepareGradient();
}
