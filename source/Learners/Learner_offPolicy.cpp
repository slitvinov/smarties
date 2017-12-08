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
  if(data->adapt_TotSeqNum <= batchSize)
    die("I do not have enough data for training. Change hyperparameters");

  //cout<<nStep<<" "<<learn_rank<<" prepareData: " <<data->Set.size()<< " "<<batchSize<<" "<<taskCounter<<" "<<nAddedGradients<<"\n"; fflush(0);
  if ( ! readyForTrain() ) {
    nAddedGradients = 0;
    return;
  }

  profiler->push_start("PRE");

  if(data->requestUpdateSamples()) {
    data->updateActiveBuffer(); //update sampling //syncDataStats
    data->shuffle_samples();
  }

  if(nStep%100==0) data->updateRewardsStats();

  taskCounter = 0;
  sequences.resize(batchSize);
  transitions.resize(batchSize);

  nAddedGradients = bSampleSequences ? data->sampleSequences(sequences) :
    data->sampleTransitions(sequences, transitions);

  profiler->stop_start("SLP");
}

bool Learner_offPolicy::readyForTrain() const
{
   //const Uint nTransitions = data->readNTransitions();
   //if(data->nSequences>=data->adapt_TotSeqNum && nTransitions<nData_b4Train())
   //  die("I do not have enough data for training. Change hyperparameters");
   //const Real nReq = std::sqrt(data->readAvgSeqLen()*16)*batchSize;
   return bTrain && data->nSequences >= nSequences4Train();
}

bool Learner_offPolicy::batchGradientReady()
{
  //if there is not enough data for training: go back to master:
  if ( ! readyForTrain() )  {
    nData_b4PolUpdates = data->readNSeen();
    return false;
  }

  const Real _nData = read_nData();
  const Real dataCounter = _nData - std::min((Real)nData_last, _nData);
  const Real stepCounter = nStep - (Real)nStep_last;
  //If I have done too many gradient steps on the avail data, go back to comm
  if( stepCounter*obsPerStep/learn_size > dataCounter ) {
    assert(unlockQueue());
    return false;
  }
  //else if threads finished processing data:
  //if(taskCounter >= batchSize) cout<<nStep<<" "<<learn_rank<<" return true with: " << nAddedGradients<< " "<<batchSize<<" "<<taskCounter<< endl; fflush(0);

  return taskCounter >= batchSize;
}

int Learner_offPolicy::spawnTrainTasks(const int availTasks) //this must be called from omp parallel region
{
  if ( !readyForTrain() ) return 0;
  //#ifdef FULLTASKING
  //  if ( !availTasks ) return 0;
  //  const int nSpawn = availTasks;
    //const int nSpawn = sequences.size();
  //#else
    const int nSpawn = sequences.size();
  //#endif
  if(sequences.size()) assert(nAddedGradients);

  if(bSampleSequences)
  {
    for (int i=0; i<nSpawn && sequences.size(); i++) {
      const Uint seq = sequences.back(); sequences.pop_back();
      addToNTasks(1);
      //#ifdef FULLTASKING
      //  #pragma omp task firstprivate(seq) if(readNTasks()<(int)nThreads)
      //#else
        #pragma omp task firstprivate(seq) //if(!availTasks)
      //#endif
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
      //#ifdef FULLTASKING
        #pragma omp task firstprivate(seq, obs) if(readNTasks()<(int)nThreads)
      //#else
        #pragma omp task firstprivate(seq, obs) //if(!availTasks)
      //#endif
      {
        const int thrID = omp_get_thread_num();
        //printf("Thread %d doing %u %u\n",thrID,sequence,transition); fflush(0);
        if(thrID == 0) profiler_ext->stop_start("WORK");
        Train(seq, obs, static_cast<Uint>(thrID));
        if(thrID == 0) profiler_ext->stop_start("COMM");
        addToNTasks(-1);
        #pragma omp atomic
        taskCounter++;
        //printf("Thread %d done %u %u\n",thrID,sequence,transition); fflush(0);
      }
    }
  }

  //#ifndef FULLTASKING
    if(!availTasks) return 0;
    #pragma omp taskwait
  //#endif
  return 0;
}

void Learner_offPolicy::prepareGradient()
{
  //cout<<nStep<<" "<<learn_rank<<" prepareGradient: " << nAddedGradients<< " "<<batchSize<<" "<<taskCounter<<" "<<data->nSequences<<endl; fflush(0);
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
  Learner::prepareGradient();
}
