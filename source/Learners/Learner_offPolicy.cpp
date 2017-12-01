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

// unlockQueue tells scheduler that has stopped receiving states from slaves
// whether should start communication again.
// for off policy learning, there is a ratio between gradient steps
// and observed transitions to be kept (approximatively) constant
bool Learner_offPolicy::unlockQueue()
{
  if( !readyForTrain() ) return true;
  const Real _nData = read_nData();
  const Real dataCounter = _nData - std::min((Real)nData_last, _nData);
  const Real stepCounter = nStep  - (Real)nStep_last;
  const Real cushionData = nSlaves;

  return dataCounter <= stepCounter*obsPerStep/learn_size +cushionData;
}
bool Learner_offPolicy::readyForAgent(const int slave)
{
  return unlockQueue(); //same: if too much data stop
}

void Learner_offPolicy::prepareData()
{
  if ( ! readyForTrain() ) return;

  profiler->push_start("PRE");

  if(nStep%100==0 || data->requestUpdateSamples())
    data->updateActiveBuffer(); //update sampling //syncDataStats

  #ifdef __CHECK_DIFF //check gradients with finite differences
    if (nStep % 100000 == 0) net->checkGrads();
  #endif

  taskCounter = 0;
  sequences.resize(batchSize);
  transitions.resize(batchSize);

  nAddedGradients = bSampleSequences ? data->sampleSequences(sequences) :
    data->sampleTransitions(sequences, transitions);

  profiler->stop_start("SLP");
}

bool Learner_offPolicy::batchGradientReady()
{
  const Real _nData = read_nData();
  const Real dataCounter = _nData - std::min((Real)nData_last, _nData);
  const Real stepCounter = nStep - (Real)nStep_last;

  //if there is not enough data for training: go back to master:
  if ( ! readyForTrain() )  {
    nData_b4PolUpdates = data->readNSeen();
    return false;
  }

  //If I have done too many gradient steps on the avail data, go back to comm
  if( stepCounter*obsPerStep/learn_size > dataCounter ) {
    //profiler_ext->stop_start("STOP");
    //printf("%g %g %g\n", stepCounter, dataCounter, _nData); fflush(0);
    assert(unlockQueue());
    return false;
  }
  //else if threads finished processing data:
  return taskCounter >= batchSize;
}

int Learner_offPolicy::spawnTrainTasks(const int availTasks) //this must be called from omp parallel region
{
  if ( !readyForTrain() ) return 0;
  #ifdef FULLTASKING
    if ( !availTasks ) return 0;
    const int nSpawn = availTasks;
    //const int nSpawn = sequences.size();
  #else
    const int nSpawn = sequences.size();
  #endif

  if(bSampleSequences)
  {
    for (int i=0; i<nSpawn && sequences.size(); i++) {
      const Uint seq = sequences.back(); sequences.pop_back();
      addToNTasks(1);
      #ifdef FULLTASKING
        #pragma omp task firstprivate(seq) if(readNTasks()<nSThreads)
      #else
        #pragma omp task firstprivate(seq) //if(!availTasks)
      #endif
      {
        const int thrID = omp_get_thread_num();
        //printf("Thread %d doing %u\n",thrID,sequence); fflush(0);
        if(thrID == 0) profiler_ext->stop_start("WORK");
        Train_BPTT(seq, static_cast<Uint>(thrID));
        if(thrID == 0) profiler_ext->stop_start("COMM");
        addToNTasks(-1);
        #pragma omp atomic
        taskCounter++;
        //printf("Thread %d done with %u\n",thrID,sequence); fflush(0);
      }
    }
  }
  else
  {
    for (int i=0; i<nSpawn && sequences.size(); i++) {
      const Uint seq = sequences.back(); sequences.pop_back();
      const Uint obs = transitions.back(); transitions.pop_back();
      addToNTasks(1);
      #ifdef FULLTASKING
        #pragma omp task firstprivate(seq, obs) if(readNTasks()<nSThreads)
      #else
        #pragma omp task firstprivate(seq, obs) //if(!availTasks)
      #endif
      {
        const int thrID = omp_get_thread_num();
        //printf("Thread %d doing %u %u\n",thrID,sequence,transition); fflush(0);
        if(thrID == 0) profiler_ext->stop_start("WORK");
        Train(seq, obs, static_cast<Uint>(thrID));
        if(thrID == 0) profiler_ext->stop_start("COMM");
        addToNTasks(-1);
        #pragma omp atomic
        taskCounter++;
        //printf("Thread %d done with %u %u\n",thrID,sequence,transition); fflush(0);
      }
    }
  }

  #ifndef FULLTASKING
    if(!availTasks) return 0;
    #pragma omp taskwait
  #endif
  return 0;
}

void Learner_offPolicy::applyGradient()
{
  if(! nAddedGradients) {
    nData_last = 0;
    nStoredSeqs_last = data->nSequences;
  } else {
    //shift data / gradient counters to maintain grad stepping to sample
    // collection ratio prescirbed by obsPerStep
    const Real stepCounter = nStep+1 - (Real)nStep_last;
    nData_last += stepCounter*obsPerStep/learn_size;
    nStep_last = nStep + 1; //actual counter advanced by base class
    assert(taskCounter == batchSize);
  }

  Learner::applyGradient();
}

/*
void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
{
  screenOut<<" DKL:["<<DKL_target<<" "<<penalDKL<<"] prec:"<<Qprec
      <<" polStats:["<<print(opcInfo->avgVec[0])<<"]";
  fileOut<<" "<<print(opcInfo->avgVec[0])<<" "<<print(opcInfo->stdVec[0]);
}
*/
