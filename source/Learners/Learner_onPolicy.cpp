/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner_onPolicy.h"

int Learner_onPolicy::spawnTrainTasks(const int availTasks)
{
  return 0;
} //do nothing

void Learner_onPolicy::prepareData() //cannot call from omp parallel region
{
  return;
}

bool Learner_onPolicy::batchGradientReady() //are all workspaces filled?
{
  bool done = true;
  for (Uint i = 0; i < work.size() && done; i++)
    done = done && work[i]->done;
  if(!done) return false;

  #pragma omp taskwait
  return true;
}

bool Learner_onPolicy::readyForAgent(const int slave, const int agent)
{
  int ret = retrieveAssignment(agent);

  if(ret<0) //then i can assign an other workspace
  {
    int avail = checkFirstAvailable();
    if(avail<0) return false; //all workspace taken, wait for ones in progress

    assert(avail%nAgentsPerSlave == 0); //assignment is done slave-wise
    const int islave = agent/nAgentsPerSlave; //actually slave-1

    for(Uint i=islave*nAgentsPerSlave; i<(islave+1)*nAgentsPerSlave; i++)
    {
      #ifndef NDEBUG
      if(retrieveAssignment(i) >= 0)
        die("FATAL Starting a new agent before terminating all others on a given slave with an on-policy algo is not supported.\n");
      if(static_cast<int>(i)==agent) ret = avail;
      #endif
      assert(avail < static_cast<int>(work.size()));
      assert(work[avail]->series.size() == 0);
      assert(work[avail]->agent==-1 && work[avail]->done==0);
      work[avail++]->agent = i; //assign
      assert(retrieveAssignment(i) == avail-1);
    };
    assert(ret>=0);
    assert(retrieveAssignment(agent) == ret);
    return true;
  }
  else
  return true;
}

void Learner_onPolicy::applyGradient()//cannot be called from omp parallel
{
  #ifndef NDEBUG
  Uint nAddedGradients_test = 0;
  for (Uint i = 0; i < work.size(); i++)
    nAddedGradients_test += work[i]->series.size()-1;
  #endif
  assert(nAddedGradients_test == nAddedGradients);
  if(!nAddedGradients) return; //then this was called WITHOUT a batch ready

  profiler->stop_start("UPW");
  dataUsage += nAddedGradients;
  batchUsage++;

  stackAndUpdateNNWeights();

  if(opt->nepoch%100 ==0) processStats();

  profiler->stop_all();

  if(opt->nepoch%100==0 && !learn_rank) {
    profiler->printSummary();
    profiler->reset();
  }

  nAddedGradients = 0;
  assert(work.size() == batchSize);
  for (Uint i = 0; i < work.size(); i++) work[i]->clear();

  //initial assignment of workspaces needed by processing queue:
  for (Uint i = 0; i < nAgents; i++) work[i]->agent = i;
}

bool Learner_onPolicy::slaveHasUnfinishedSeqs(const int slave) const
{
  for(Uint i=slave*nAgentsPerSlave; i<(slave+1)*nAgentsPerSlave; i++)
    if(retrieveAssignment(i)>=0)
      return true;
  return false;
}
