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
_s.minTotObsNum>_s.batchSize? _s.minTotObsNum : _s.maxTotObsNum) {}

void Learner_offPolicy::prepareData()
{
  // it should be impossible to get here before starting batch update was ready.
  if(updateComplete) die("undefined behavior");

  // then no need to prepare a new update
  if(updatePrepared) return;


  if( not readyForTrain() ) return; // Do not prepare an update

  profiler->stop_start("PRE");
  if(nStep%1000==0) data->updateRewardsStats();
  taskCounter = 0;
  samp_seq = vector<Uint>(batchSize, -1);
  samp_obs = vector<Uint>(batchSize, -1);
  if(bSampleSequences) data->sampleSequences(samp_seq);
  else data->sampleTransitions_OPW(samp_seq, samp_obs);
  updatePrepared = true;
  profiler->stop_start("SLP");
}

bool Learner_offPolicy::readyForTrain() const
{
  //const Uint nTransitions = data->readNTransitions();
  //if(data->nSequences>=data->adapt_TotSeqNum && nTransitions<nData_b4Train())
  //  die("I do not have enough data for training. Change hyperparameters");
  //const Real nReq = std::sqrt(data->readAvgSeqLen()*16)*batchSize;
  const bool ready = bTrain && data->readNData() >= nObsPerTraining;
  if(not ready && bTrain && omp_get_thread_num() == 0 && !learn_rank) {
    const Uint currPerc = data->readNData() * 100. / (Real) nObsPerTraining;
    if(currPerc>percData) {
      percData = currPerc;
      printf("\rCollected %d%% of data required to begin training. ", percData);
    }
  }

   return ready;
}

bool Learner_offPolicy::stopGrads() const
{
  //stop grads if the ratio of observed trantition to steps is above
  //user defined ratio `obsPerStep`
  const Real _nData = (Real)data->readNSeen() - nData_b4Startup;
  const Real dataCounter = _nData - (Real)nData_last;
  const Real stepCounter =  nStep - (Real)nStep_last;
  // lock the queue if we have more data than grad step ratio
  const bool notEnoughData = dataCounter < stepCounter*obsPerStep;
  return notEnoughData;
}

bool Learner_offPolicy::lockQueue() const
{
  // lockQueue tells scheduler that has stopped receiving states from slaves
  // whether should start communication again.
  // for off policy learning, there is a ratio between gradient steps
  // and observed transitions to be kept (approximatively) constant

  if( not readyForTrain() ) {
    //if there is not enough data for training, need more data
    assert( not updatePrepared );
    return false;
  }

  if( not updatePrepared ) { //then was waiting for data without an update ready
    return true; // need to exit data loop to prepare an update
  }

  //const Real _nData = (Real)data->readNConcluded() - nData_b4Startup;
  const Real _nData = data->readNSeen() - nData_b4Startup;
  const Real dataCounter = _nData - (Real)nData_last;
  const Real stepCounter =  nStep - (Real)nStep_last;
  // Lock the queue if we have !added to the training set! more observations
  // than (grad_step * obsPerStep) or if the update is ready.
  // The distinction between "added to set" and "observed" allows removing
  // some load inbalance, with only has marginal effects on algorithms.
  // Load imb. is reduced by minimizing pauses in either data or grad stepping.
  const bool tooMuchData = dataCounter > stepCounter*obsPerStep;
  //return tooMuchData || batchComplete();
  return tooMuchData;
}

void Learner_offPolicy::spawnTrainTasks_par()
{
  if(updateComplete) die("undefined behavior");
  if( not updatePrepared ) return;
  //if( stopGrads() ) { // postpone
  //  #pragma omp task
  //  spawnTrainTasks_par();
  //}
  if( bSampleSequences && data->nSequences < batchSize)
    die("Parameter maxTotObsNum is too low for given problem");

  for (Uint i=0; i<batchSize; i++)
  {
    Uint seq = samp_seq[i], obs = samp_obs[i];
    #pragma omp task firstprivate(seq, obs)
    {
      const int thrID = omp_get_thread_num();
      if(thrID == 0) profiler_ext->stop_start("WORK");
      //printf("Thread %d done %u %u %f\n",thrID,seq,obs,data->Set[seq]->offPol_weight[obs]); fflush(0);
      if(bSampleSequences)
      {
        obs = data->Set[seq]->ndata()-1;
        Train_BPTT(seq, thrID);
        #pragma omp atomic
        nAddedGradients += data->Set[seq]->ndata();
      }
      else
      {
        //data->sampleTransition(seq, obs, thrID);
        Train(seq, obs, thrID);
        #pragma omp atomic
        nAddedGradients++;
      }

      #pragma omp atomic
        taskCounter++;
      input->gradient(thrID);
      data->Set[seq]->setSampled(obs);

      if(thrID==0) profiler_ext->stop_start("SLP");
      if(thrID==1) profiler->stop_start("SLP");
    }
  }
  samp_seq.clear();
  samp_obs.clear();
}

void Learner_offPolicy::spawnTrainTasks_seq()
{
  if(taskCounter >= batchSize) updateComplete = true;

  if(not updatePrepared) nData_b4Startup = data->readNConcluded();
}

void Learner_offPolicy::prepareGradient()
{
  const bool bWasPrepareReady = updateComplete;

  Learner::prepareGradient();

  if(bWasPrepareReady) {
    profiler->stop_start("PRNE");
    advanceCounters();
    data->prune(CmaxRet, FILTER_ALGO);
    profiler->stop_start("SLP");
  }
}
