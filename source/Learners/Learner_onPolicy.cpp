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
    // data->clearAll();
    cntKept = data->clearOffPol(CmaxPol, 0.1);
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
  return bTrain && data->readNData() >= nHorizon + cntKept;
}

void Learner_onPolicy::spawnTrainTasks_seq()
{
  if( updateComplete ) die("undefined behavior");
  if( data->readNData() < nHorizon ) die("undefined behavior");
  if( not bTrain ) return;
  if(nStep==0) data->updateRewardsStats(nStep, 1);
  updatePrepared = true;
  vector<Uint> samp_seq(batchSize, -1), samp_obs(batchSize, -1);
  data->sampleTransitions_OPW(samp_seq, samp_obs);

  #pragma omp parallel for schedule(dynamic)
  for (Uint i=0; i<batchSize; i++)
  {
    const int thrID = omp_get_thread_num();
    if(thrID==0) profiler_ext->stop_start("WORK");
    const Uint seq = samp_seq[i], obs = samp_obs[i];
    Train(seq, obs, thrID);
    input->gradient(thrID);
    data->Set[seq]->setSampled(obs);
    if(thrID==1) profiler->stop_start("SLP");
    if(thrID==0) profiler_ext->stop_start("SLP");
  }
  updateComplete = true;
}

void Learner_onPolicy::spawnTrainTasks_par() { }

void Learner_onPolicy::prepareGradient()
{
  if (updateComplete && bTrain) {
    cntBatch += batchSize;
    if(cntBatch >= nHorizon) {
      data->updateRewardsStats(nStep, 0.001);
      cntBatch = 0;
      cntEpoch++;
    }
  }
  Learner::prepareGradient();
}
