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
Learner(_env,_s), nObsPerTraining(_s.maxTotObsNum/_s.learner_size),
obsPerStep_orig(_s.obsPerStep) { }

void Learner_offPolicy::prepareData()
{
  if(data->adapt_TotSeqNum <= batchSize)
    die("I do not have enough data for training. Change hyperparameters")

  // it should be impossible to get here before starting batch update was ready.
  if(updateComplete) die("undefined behavior");

  // then no need to prepare a new update
  if(updatePrepared) return;

  profiler->stop_start("PRE");
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

  if(data->requestUpdateSamples())
    data->updateActiveBuffer(); //update sampling //syncDataStats

  if(nStep%100==0) data->updateRewardsStats();

  taskCounter = 0;
  updatePrepared = true;
  nToSpawn = batchSize;
  profiler->stop_start("SLP");
}

bool Learner_offPolicy::readyForTrain() const
{
   //const Uint nTransitions = data->readNTransitions();
   //if(data->nSequences>=data->adapt_TotSeqNum && nTransitions<nData_b4Train())
   //  die("I do not have enough data for training. Change hyperparameters");
   //const Real nReq = std::sqrt(data->readAvgSeqLen()*16)*batchSize;
   return bTrain && data->nTransitions >= nSequences4Train();
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

  for (Uint i=0; i<nToSpawn; i++)
  {
    addToNTasks(1);
    #pragma omp task
    {
      Uint seq, obs;
      const int thrID = omp_get_thread_num();
      //printf("Thread %d doing %u %u\n",thrID,seq,obs); fflush(0);
      if(thrID == 0) profiler_ext->stop_start("WORK");

      if(bSampleSequences) {
        data->sampleSequence(seq, thrID);
        Train_BPTT(seq, thrID);
        #pragma omp atomic
        nAddedGradients += data->Set[seq]->ndata();
      }
      else                 {
        data->sampleTransition(seq, obs, thrID);
        Train(seq, obs, thrID);
        #pragma omp atomic
        nAddedGradients++;
      }

      if(thrID == 0) profiler_ext->stop_start("COMM");
      #pragma omp atomic
      taskCounter++;
      //printf("Thread %d done %u %u\n",thrID,seq,obs); fflush(0);
      if(taskCounter >= batchSize) updateComplete = true;
      addToNTasks(-1);
    }
  }
  nToSpawn = 0;
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

bool Learner_offPolicy::predefinedNetwork(Builder& input_net)
{
  return false;
}
