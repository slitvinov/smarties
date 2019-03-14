//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_approximator.h"
#include "../Network/Approximator.h"
#include "../Network/Builder.h"
#include <chrono>

Learner_approximator::Learner_approximator(Environment*const E, Settings&S): Learner(E, S), input(new Encapsulator("input", S, data))
{
  if(input->nOutputs() == 0) return;
  Builder input_build(S);
  input_build.addInput( input->nOutputs() );
  bool builder_used = env->predefinedNetwork(input_build);
  if(builder_used) {
    Network* net = input_build.build(true);
    Optimizer* opt = input_build.opt;
    input->initializeNetwork(net, opt);
  }
  if(not bSampleSequences && nObsB4StartTraining < batchSize)
    die("Parameter minTotObsNum is too low for given problem");
}
Learner_approximator::~Learner_approximator() {
  _dispose_object(input);
}

void Learner_approximator::spawnTrainTasks()
{
  if(bSampleSequences && data->readNSeq() < batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");

  std::vector<Uint> samp_seq = std::vector<Uint>(batchSize, -1);
  std::vector<Uint> samp_obs = std::vector<Uint>(batchSize, -1);
  data->sample(samp_seq, samp_obs);

  for(Uint i=0; i<batchSize && bSampleSequences; i++)
    assert( samp_obs[i] == data->get(samp_seq[i])->ndata() - 1 );

  if(bSampleSequences) {
  #pragma omp parallel for collapse(2) schedule(static) num_threads(nThreads)
    for (Uint wID=0; wID<ESpopSize; wID++)
      for (Uint bID=0; bID<batchSize; bID++) {
        const Uint thrID = omp_get_thread_num();
        TrainBySequences(samp_seq[bID], wID, bID, thrID);
        input->gradient(thrID);
      }
  } else {
  static const Uint CS = batchSize / nThreads;
  #pragma omp parallel for collapse(2) schedule(static,CS) num_threads(nThreads)
    for (Uint wID=0; wID<ESpopSize; wID++)
      for (Uint bID=0; bID<batchSize; bID++) {
        const Uint thrID = omp_get_thread_num();
        Train(samp_seq[bID], samp_obs[bID], wID, bID, thrID);
        input->gradient(thrID);
      }
  }
}

void Learner_approximator::prepareGradient()
{
  const Uint currStep = nGradSteps()+1;

  profiler->stop_start("ADDW");
  for(auto & net : F) net->prepareUpdate();
  input->prepareUpdate();

  for(auto & net : F) net->updateGradStats(learner_name, currStep-1);
}

void Learner_approximator::applyGradient()
{
  profiler->stop_start("GRAD");
  for(auto & net : F) net->applyUpdate();
  input->applyUpdate();
}

void Learner_approximator::getMetrics(ostringstream& buf) const
{
  Learner::getMetrics(buf);
  input->getMetrics(buf);
  for(auto & net : F) net->getMetrics(buf);
}
void Learner_approximator::getHeaders(ostringstream& buf) const
{
  Learner::getHeaders(buf);
  input->getHeaders(buf);
  for(auto & net : F) net->getHeaders(buf);
}

void Learner_approximator::restart()
{
  if(settings.restart == "none") return;
  if(!learn_rank) printf("Restarting from saved policy...\n");

  for(auto & net : F) net->restart(settings.restart+"/"+learner_name);
  input->restart(settings.restart+"/"+learner_name);

  for(auto & net : F) net->save("restarted_"+learner_name, false);
  input->save("restarted_"+learner_name, false);

  Learner::restart();

  if(input->opt not_eq nullptr) input->opt->nStep = _nGradSteps;
  for(auto & net : F) net->opt->nStep = _nGradSteps;
}

void Learner_approximator::save()
{
  const Uint currStep = nGradSteps()+1;
  const Real freqSave = freqPrint * PRFL_DMPFRQ;
  const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  const bool bBackup = currStep % freqBackup == 0;
  for(auto & net : F) net->save(learner_name, bBackup);
  input->save(learner_name, bBackup);

  Learner::save();
}

//TODO: generalize!!
bool Learner_approximator::predefinedNetwork(Builder& input_net, Uint privateNum)
{
  bool ret = false; // did i add layers to input net?
  if(input_net.nOutputs > 0) {
     input_net.nOutputs = 0;
     input_net.layers.back()->bOutput = false;
     warn("Overwritten ENV's specification of CNN to add shared layers");
  }
  vector<int> sizeOrig = settings.readNetSettingsSize();
  while ( sizeOrig.size() != privateNum )
  {
    const int size = sizeOrig[0];
    sizeOrig.erase(sizeOrig.begin(), sizeOrig.begin()+1);
    const bool bOutput = sizeOrig.size() == privateNum;
    input_net.addLayer(size, settings.nnFunc, bOutput);
    ret = true;
  }
  settings.nnl1 = sizeOrig.size() > 0? sizeOrig[0] : 0;
  settings.nnl2 = sizeOrig.size() > 1? sizeOrig[1] : 0;
  settings.nnl3 = sizeOrig.size() > 2? sizeOrig[2] : 0;
  settings.nnl4 = sizeOrig.size() > 3? sizeOrig[3] : 0;
  settings.nnl5 = sizeOrig.size() > 4? sizeOrig[4] : 0;
  settings.nnl6 = sizeOrig.size() > 5? sizeOrig[5] : 0;
  return ret;
}

void Learner_approximator::createSharedEncoder(const Uint privateNum)
{
  if(input->net not_eq nullptr) {
    delete input->opt; input->opt = nullptr;
    delete input->net; input->net = nullptr;
  }
  if(input->nOutputs() == 0) return;
  Builder input_build(settings);
  bool bInputNet = false;
  input_build.addInput( input->nOutputs() );
  bInputNet = bInputNet || env->predefinedNetwork(input_build);
  bInputNet = bInputNet || predefinedNetwork(input_build, privateNum);
  if(bInputNet) {
    Network* net = input_build.build(true);
    input->initializeNetwork(net, input_build.opt);
  }
}

//bool Learner::predefinedNetwork(Builder & input_net)
//{
//  return false;
//}
