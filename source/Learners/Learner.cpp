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
  if(not updatePrepared) {
    profiler_ext->stop_all();
    profiler->stop_all();
    profiler->reset();
    profiler_ext->reset();
    profiler->stop_start("SLP");
    profiler_ext->stop_start("SLP");
  }

  if(not updateComplete)
    return; //then this was called WITHOUT a batch ready

  assert(updatePrepared);
  // Learner is ready for the update: send the task to the networks and
  // start preparing the next one
  updatePrepared = false;
  updateComplete = false;

  profiler->stop_start("ADDW");
  for(auto & net : F) net->prepareUpdate();
  input->prepareUpdate();

  nStep++;

  for(auto & net : F) net->updateGradStats(nStep);

  profiler->stop_all();

  if(nStep%(1000*PRFL_DMPFRQ)==0 && !learn_rank) {
    profiler->printSummary();
    profiler->reset();

    profiler_ext->stop_all();
    profiler_ext->printSummary();
    profiler_ext->reset();

    for(auto & net : F) net->save(learner_name);
    input->save(learner_name);
  }

  if(nStep%1000 ==0) {
    profiler->stop_start("STAT");
    processStats();
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
  ostringstream buf;
  stats.getMetrics(buf);
  data->getMetrics(buf);
  for(auto & net : F) net->getMetrics(buf);
  getMetrics(buf);

  if(learn_rank) return;

  FILE* fout = fopen ("stats.txt","a");

  ostringstream head;
  if( nStep%(1000*PRFL_DMPFRQ)==0 || nStep==1000 ) {
    stats.getHeaders(head);
    data->getHeaders(head);
    for(auto & net : F) net->getHeaders(head);
    getHeaders(head);
    printf("#/1e3 %s\n", head.str().c_str());
    if(nStep==1000)
      fprintf(fout, "#/1e3 %s\n", head.str().c_str());
  }

  fprintf(fout, "%05d %s\n", (int)nStep/1000, buf.str().c_str());
  printf("%05d %s\n", (int)nStep/1000, buf.str().c_str());
  fclose(fout);
  fflush(0);
}

void Learner::getMetrics(ostringstream& buf) const {}
void Learner::getHeaders(ostringstream& buf) const {}

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

//TODO: generalize!!
bool Learner::predefinedNetwork(Builder& input_net, Settings& settings)
{
  if(settings.nnl2<=0) return false;

  if(input_net.nOutputs > 0) {
    input_net.nOutputs = 0;
    input_net.layers.back()->bOutput = false;
  }
  //                 size       function     is output (of input net)?
  input_net.addLayer(settings.nnl1, settings.nnFunc, settings.nnl3<=0);
  settings.nnl1 = settings.nnl2;
  if(settings.nnl3>0) {
    input_net.addLayer(settings.nnl2, settings.nnFunc, settings.nnl4<=0);
    settings.nnl1 = settings.nnl3;
    if(settings.nnl4>0) {
      input_net.addLayer(settings.nnl3, settings.nnFunc, settings.nnl5<=0);
      settings.nnl1 = settings.nnl4;
      if(settings.nnl5>0) {
        input_net.addLayer(settings.nnl4, settings.nnFunc, settings.nnl6<=0);
        settings.nnl1 = settings.nnl5;
        if(settings.nnl6>0) {
          input_net.addLayer(settings.nnl5, settings.nnFunc, true);
          settings.nnl1 = settings.nnl6;
        }
      }
    }
  }
  settings.nnl2 = 0; // value, adv and pol nets will be one-layer
  return true;
}

//bool Learner::predefinedNetwork(Builder & input_net)
//{
//  return false;
//}
