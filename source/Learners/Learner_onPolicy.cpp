/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner_onPolicy.h"

Learner_onPolicy::Learner_onPolicy(Environment*const _env, Settings&_s): Learner(_env, _s) { }

// unlockQueue tells scheduler that has stopped receiving states from slaves
// whether should start communication again.
// for on policy learning, when enough data is collected from slaves
// gradient stepping starts and collection is paused
// when training is concluded collection restarts
bool Learner_onPolicy::unlockQueue()
{
  if(cntEpoch == nEpochs) {
    for(auto& old_traj: data->Set) //delete already-used trajectories
      _dispose_object(old_traj);
    //for(auto& old_traj: data->inProgress)
    //  old_traj->clear();//remove from in progress: now off policy
    data->Set.clear(); //clear trajectories used for learning
    //reset batch learning counters
    cntTrajectories = 0; cntHorizon = 0; cntEpoch = 0; cntBatch = 0;
    return true;
  }
  return false;
}
bool Learner_onPolicy::readyForAgent(const int slave)
{
  //block creation if we have reached enough data for a batch
  if(cntHorizon>=nHorizon) return false;
  else return true;
}

void Learner_onPolicy::prepareData()
{
  #ifdef __CHECK_DIFF //check gradients with finite differences
    if (nStep % 100000 == 0) net->checkGrads();
  #endif
  return;
}

bool Learner_onPolicy::batchGradientReady()
{
  return nAddedGradients == batchSize;
}

int Learner_onPolicy::spawnTrainTasks(const int availTasks)
{
  if ( cntHorizon < nHorizon  || ! bTrain) return 0;
  vector<Uint> sequences(batchSize), transitions(batchSize);
  nAddedGradients = data->sampleTransitions(sequences, transitions);

  #pragma omp parallel for schedule(dynamic)
  for (Uint i=0; i<batchSize; i++)
      Train(sequences[i], transitions[i], omp_get_thread_num());
  return 0;
}

void Learner_onPolicy::applyGradient()
{
  if (nAddedGradients && bTrain) {
    cntBatch += batchSize;
    if(cntBatch >= nHorizon) {
      cntBatch = 0;
      cntEpoch++;
    }
  }
  Learner::applyGradient();
}
