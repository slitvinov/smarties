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
Learner(_env,_s), obsPerStep_orig(_s.obsPerStep), nObsPerTraining(
(_s.minTotObsNum>_s.batchSize? _s.minTotObsNum : _s.maxTotObsNum)/learn_size) {}

void Learner_offPolicy::prepareData()
{
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
    nData_last = 0;
  }

  // Do not prepare an update:
  if(    waitingForData || not readyForTrain()) {
    waitingForData = true;
    return;
  }

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
  const Real dataCounter = _nData - (Real)nData_last;
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
  const Real dataCounter = _nData - (Real)nData_last;
  const Real stepCounter = nStep  - (Real)nStep_last;
  // cushion allows tolerance to collect a bit more data than strictly necessary
  // to avoid bottlenecks, but no significant effect on learning
  return stepCounter*obsPerStep/learn_size >= dataCounter;
}

int Learner_offPolicy::spawnTrainTasks()
{
  if( updateComplete || not updatePrepared || nToSpawn == 0) return 0;
  if(bSampleSequences && data->nSequences<nToSpawn)
    die("Parameter maxTotObsNum is too low for given problem");

  vector<Uint> samp_seq = bSampleSequences? data->sampleSequences(nToSpawn) : vector<Uint>(nToSpawn);
  for (Uint i=0; i<nToSpawn; i++)
  {
    addToNTasks(1);
    Uint seq = samp_seq[i], obs = -1;
    #pragma omp task firstprivate(seq, obs)
    {
      const int thrID = omp_get_thread_num();
      if(thrID == 0) profiler_ext->stop_start("WORK");
      //printf("Thread %d done %u %u\n",thrID,seq,obs); fflush(0);
      if(bSampleSequences) {
        obs = data->Set[seq]->ndata()-1;
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
      input->gradient(thrID);
      data->Set[seq]->setSampled(obs);

      if(thrID == 0) profiler_ext->stop_start("COMM");
      #pragma omp atomic
      taskCounter++;
      if(taskCounter >= batchSize) updateComplete = true;
      addToNTasks(-1);
    }
  }
  nToSpawn = 0;
  return 0;
}

void Learner_offPolicy::prepareGradient()
{
  const bool bWasPrepareReady = updateComplete;

  Learner::prepareGradient();

  if(bWasPrepareReady) {
    nSkipped = 0;
    profiler->stop_start("PRNE");
    //shift data / gradient counters to maintain grad stepping to sample
    // collection ratio prescirbed by obsPerStep
    const Real stepCounter = nStep - (Real)nStep_last;

    nData_last += stepCounter*obsPerStep/learn_size;
    nStep_last = nStep;
    data->prune(CmaxRet, ALGO);
    profiler->stop_start("SLP");
  }
}
