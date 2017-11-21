/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner_onPolicy.h"

bool Learner_onPolicy::unlockQueue()
{
  if(cntEpoch == nEpochs)
  {
    completed.clear(); //clear trajectories used for learning
    //reset batch learning counters
    cntTrajectories = cntHorizon = cntEpoch = cntBatch = 0;
    //clear workspace to store new trajectories
    for(Uint i=0; i<work.size(); i++) work[i]->clear();
    //initial assignment of workspaces needed by processing queue:
    for(Uint i = 0; i < nAgents; i++) work[i]->agent=i;
    return true;
  }
  return false;
}

void Learner_onPolicy::prepareData() //cannot call from omp parallel region
{
  return;
}

bool Learner_onPolicy::batchGradientReady() //are all workspaces filled?
{
  return nAddedGradients == batchSize;
}

void Learner_onPolicy::sampleTransitions(Uint&seq, Uint&trans, const Uint thrID)
{
  std::uniform_int_distribution<Uint> distu(0, cntHorizon-1);
  const Uint ind = distu(generators[thrID]);

  Uint k=0, back=0, indT=completed[0]->GAE.size();
  while (ind >= indT) {
    back = indT;
    indT += completed[++k]->GAE.size();
  }

  seq = k;
  trans = ind-back;
  assert(seq<completed.size());
  assert(trans<completed[k]->GAE.size());
}

int Learner_onPolicy::spawnTrainTasks(const int availTasks)
{
  if ( cntHorizon < nHorizon  || ! bTrain) return 0;

  for (Uint i=0; i<batchSize; i++) {
    #pragma omp task
    {
      Uint seq=0, trans=0;
      const int thrID = omp_get_thread_num();
      sampleTransitions(seq, trans, thrID);
      Train(seq, trans, static_cast<Uint>(thrID));
    }
  }
  #pragma omp taskwait
  nAddedGradients = batchSize;
  return 0;
}

bool Learner_onPolicy::readyForAgent(const int slave)
{
  const int firstagent = (slave-1)*nAgentsPerSlave;
  int ret = retrieveAssignment(firstagent);

  if (ret>=0) {
    #ifndef NDEBUG
      for(Uint i=firstagent; i<firstagent+nAgentsPerSlave; i++)
        if(retrieveAssignment(i) < 0)
          die("Starting a new agent before terminating all others on a given slave is not supported.");
    #endif
    return true;
  }

  int avail = checkFirstAvailable();
  //all workspace taken: put comm in queue while grad desc in progress
  if(avail<0) return false;

  assert(avail%nAgentsPerSlave == 0); //assignment is done slave-wise
  for(Uint i=firstagent; i<firstagent+nAgentsPerSlave; i++)
  {
    #ifndef NDEBUG
      if(retrieveAssignment(i) >= 0)
        die("Starting a new agent before terminating all others on a given slave is not supported.");
    #endif
    assert(avail < static_cast<int>(work.size()));
    assert(work[avail]->agent==-1 && work[avail]->done==0);
    work[avail++]->agent = i; //assign
    assert(retrieveAssignment(i) == avail-1);
  };
  return true;
}

void Learner_onPolicy::applyGradient()//cannot be called from omp parallel
{
  if(!nAddedGradients || ! bTrain) return; 
  //then this was called WITHOUT a batch ready

  profiler->stop_start("UPW");

  stackAndUpdateNNWeights();

  if(opt->nepoch%100 ==0) processStats();

  profiler->stop_all();

  if(opt->nepoch%1000==0 && !learn_rank) {
    profiler->printSummary();
    profiler->reset();

    profiler_ext->stop_all();
    profiler_ext->printSummary();
    profiler_ext->reset();
  }

  cntBatch += batchSize;
  nAddedGradients = 0;
  if(cntBatch >= nHorizon) {
    cntBatch = 0;
    cntEpoch++;
  }
}

bool Learner_onPolicy::slaveHasUnfinishedSeqs(const int slave) const
{
  for(Uint i=slave*nAgentsPerSlave; i<(slave+1)*nAgentsPerSlave; i++)
    if(retrieveAssignment(i)>=0) return true;
  return false;
}
