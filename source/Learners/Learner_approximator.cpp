//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learner_approximator.h"
#include "Network/Approximator.h"
#include "Network/Builder.h"
#include <chrono>

namespace smarties
{

Learner_approximator::Learner_approximator(MDPdescriptor& MDP_,
                                           Settings& S_,
                                           DistributionInfo& D_) :
                                           Learner(MDP_, S_, D_)
{
  if(!settings.bSampleSequences && nObsB4StartTraining<(long)settings.batchSize)
    die("Parameter minTotObsNum is too low for given problem");
}

Learner_approximator::~Learner_approximator()
{
  for(auto & net : networks) {
    if(net not_eq nullptr) {
      delete net;
      net = nullptr;
    }
  }
}

void Learner_approximator::spawnTrainTasks()
{
  if(settings.bSampleSequences && data->readNSeq() < (long) settings.batchSize)
    die("Parameter minTotObsNum is too low for given problem");

  profiler->stop_start("SAMP");

  const Uint nThr = distrib.nThreads, CS =  settings.batchSize / nThr;
  const Uint batchSize = settings.batchSize, ESpopSize = settings.ESpopSize;
  const MiniBatch MB = data->sampleMinibatch(batchSize, nGradSteps() );

  if(settings.bSampleSequences)
  {
    #pragma omp parallel for collapse(2) schedule(dynamic,1) num_threads(nThr)
    for (Uint wID=0; wID<ESpopSize; ++wID)
    for (Uint bID=0; bID<batchSize; ++bID) {
      Train(MB, wID, bID);
      // backprop, from last net to first for dependencies in gradients:
      for (const auto & net : Utilities::reverse(networks) ) net->backProp(bID);
    }
  }
  else
  {
    #pragma omp parallel for collapse(2) schedule(static,CS) num_threads(nThr)
    for (Uint wID=0; wID<ESpopSize; ++wID)
    for (Uint bID=0; bID<batchSize; ++bID) {
      Train(MB, wID, bID);
      // backprop, from last net to first for dependencies in gradients:
      for (const auto & net : Utilities::reverse(networks) ) net->backProp(bID);
    }
  }
}

void Learner_approximator::prepareGradient()
{
  const Uint currStep = nGradSteps()+1;

  profiler->stop_start("ADDW");
  for(const auto & net : networks) {
    net->prepareUpdate();
    net->updateGradStats(learner_name, currStep-1);
  }
}

void Learner_approximator::applyGradient()
{
  profiler->stop_start("GRAD");
  for(const auto & net : networks) net->applyUpdate();
}

void Learner_approximator::getMetrics(std::ostringstream& buf) const
{
  Learner::getMetrics(buf);
  for(const auto & net : networks) net->getMetrics(buf);
}
void Learner_approximator::getHeaders(std::ostringstream& buf) const
{
  Learner::getHeaders(buf);
  for(const auto & net : networks) net->getHeaders(buf);
}

void Learner_approximator::restart()
{
  if(settings.restart == "none") return;
  if(!learn_rank) printf("Restarting from saved policy...\n");

  for(const auto & net : networks) net->restart(settings.restart+"/"+learner_name);
  for(const auto & net : networks) net->save("restarted_"+learner_name, false);

  Learner::restart();

  for(const auto & net : networks) net->setNgradSteps(_nGradSteps);
}

void Learner_approximator::save()
{
  const Uint currStep = nGradSteps()+1;
  const Real freqSave = freqPrint * PRFL_DMPFRQ;
  const Uint freqBackup = std::ceil(settings.saveFreq / freqSave)*freqSave;
  const bool bBackup = currStep % freqBackup == 0;
  for(const auto & net : networks) net->save(learner_name, bBackup);

  Learner::save();
}

// Create preprocessing network, which contains conv layers requested by env
// and some shared fully connected layers, read from vector nnLayerSizes
// from settings. The last privateLayersNum sizes of nnLayerSizes are not
// added here because we assume those sizes will parameterize the approximators
// that take the output of the preprocessor and produce policies,values, etc.
bool Learner_approximator::createEncoder(Sint privateLayersNum)
{
  const Sint totLayersNum = settings.nnLayerSizes.size();
  // If privateLayersNum defaults to -1, assumed that all are private. Meaning
  // that each approximator will consists of all nnLayerSizes.
  if(privateLayersNum<0) privateLayersNum = totLayersNum;
  if(privateLayersNum>totLayersNum) privateLayersNum = totLayersNum;

  const Uint nPreProcLayers = totLayersNum - privateLayersNum;
  const std::vector<Uint> origLayers = settings.nnLayerSizes;
  std::vector<Uint> preprocessingLayers = settings.nnLayerSizes;
  preprocessingLayers.resize(nPreProcLayers); // take first nPreProcLayers

  // remaining layer sizes in nnLayerSizes will be used by other approximators:
  settings.nnLayerSizes.clear();
  for(Sint i=nPreProcLayers; i<totLayersNum; ++i)
    settings.nnLayerSizes.push_back( origLayers[i] );
  assert( (Uint) privateLayersNum == settings.nnLayerSizes.size() );

  if ( MDP.conv2dDescriptors.size() == 0 and nPreProcLayers == 0 )
    return false; // no preprocessing

  if(networks.size()>0) warn("some network was created before preprocessing");
  networks.push_back( new Approximator( "encoder", settings, distrib, data ) );
  networks.back()->buildPreprocessing(preprocessingLayers);

  return true;
}

void Learner_approximator::initializeApproximators()
{
  for(const auto& net : networks)
  {
    net->initializeNetwork();
  }
}

}

/*

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
, input(new Encapsulator("input", S_, data))
if(input->nOutputs() == 0) return;
Builder input_build(S);
input_build.addInput( input->nOutputs() );
bool builder_used = env->predefinedNetwork(input_build);
if(builder_used) {
  Network* net = input_build.build(true);
  Optimizer* opt = input_build.opt;
  input->initializeNetwork(net, opt);
}
*/
