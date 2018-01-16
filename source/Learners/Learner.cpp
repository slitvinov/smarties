/*
 *  Learner.cpp
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Learner.h"
#include "../Network/Builder.h"
#include <chrono>

Learner::Learner(Environment*const _env, Settings & _s) :
mastersComm(_s.mastersComm), env(_env), bSampleSequences(_s.bSampleSequences),
bTrain(_s.bTrain), nAgents(_s.nAgents), batchSize(_s.batchSize),
totNumSteps(_s.totNumSteps), nThreads(_s.nThreads), nSlaves(_s.nSlaves),
policyVecDim(_s.policyVecDim), greedyEps(_s.greedyEps), epsAnneal(_s.epsAnneal),
gamma(_s.gamma), CmaxPol(_s.impWeight), learn_rank(_s.learner_rank),
learn_size(_s.learner_size), aInfo(env->aI), sInfo(env->sI),
generators(_s.generators), Vstats(nThreads), nTasks(_s.global_tasking_counter)
{
  assert(nThreads>1);
  if(bSampleSequences) printf("Sampling sequences.\n");
  profiler = new Profiler();
  data = new MemoryBuffer(env, _s);
  input = new Encapsulator("input", _s, data);
  profiler->stop_start("SLP");

  Builder input_build(_s);
  input_build.addInput( input->nOutputs() );
  bool builder_used = env->predefinedNetwork(input_build);
  if(builder_used) {
    Network* net = input_build.build();
    Optimizer* opt = input_build.opt;
    input->initializeNetwork(net, opt);
  }
}

void Learner::clearFailedSim(const int agentOne, const int agentEnd)
{
  data->clearFailedSim(agentOne, agentEnd);
}

void Learner::pushBackEndedSim(const int agentOne, const int agentEnd)
{
  data->pushBackEndedSim(agentOne, agentEnd);
}

void Learner::prepareGradient() //this cannot be called from omp parallel region
{
  if(not updateComplete)
    return; //then this was called WITHOUT a batch ready

  assert(updatePrepared && not waitingForData);
  // Learner is ready for the update: send the task to the networks and
  // start preparing the next one
  updatePrepared = false;
  updateComplete = false;

  profiler->stop_start("ADDW");
  for(auto & net : F) net->prepareUpdate();
  input->prepareUpdate();

  nStep++;

  if(nStep%100 ==0) {
    profiler->stop_start("STAT");
    processStats();
  }
  profiler->stop_all();

  if(nStep%1000==0 && !learn_rank) {
    profiler->printSummary();
    profiler->reset();

    profiler_ext->stop_all();
    profiler_ext->printSummary();
    profiler_ext->reset();

    for(auto & net : F) net->save(learner_name);
    input->save(learner_name);
  }
  profiler->stop_start("SLP");
}

void Learner::synchronizeGradients()
{
  profiler->stop_start("UPW");
  for(auto & net : F) net->applyUpdate();
  input->applyUpdate();
  profiler->stop_start("SLP");
}

void Learner::processStats()
{
  stats.reduce(Vstats);
  ostringstream fileOut, screenOut;
  stats.getMetrics(fileOut, screenOut);
  data->getMetrics(fileOut, screenOut);
  for(auto & net : F) net->getMetrics(fileOut, screenOut);
  getMetrics(fileOut, screenOut);

  if(learn_rank) return;
  ofstream fout; fout.open("stats.txt", ios::app);
  fout<<fileOut.str()<<endl; fout.flush(); fout.close();
  printf("%lu %s\n", nStep, screenOut.str().c_str()); fflush(0);
}

void Learner::getMetrics(ostringstream&fileOut,ostringstream&screenOut) const {}

void Learner::restart()
{
  if(!learn_rank) printf("Restarting from saved policy...\n");

  for(auto & net : F) net->restart();
  input->restart();
  data->restart();

  for(auto & net : F) net->save("restarted_");
  input->save("restarted_");
}

bool Learner::slaveHasUnfinishedSeqs(const int slave) const
{
  const Uint nAgentsPerSlave = env->nAgentsPerRank;
  for(Uint i=slave*nAgentsPerSlave; i<(slave+1)*nAgentsPerSlave; i++)
    if(data->inProgress[i]->tuples.size()) return true;
  return false;
}

//bool Learner::predefinedNetwork(Builder & input_net)
//{
//  return false;
//}
