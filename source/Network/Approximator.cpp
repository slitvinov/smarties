//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Approximator.h"
#include "Optimizer.h"
#include "Network.h"

namespace smarties
{

Approximator::Approximator(std::string name_,
                           const Settings&S,
                           const DistributionInfo&D,
                           const MemoryBuffer* const replay_,
                           const Approximator* const preprocessing_,
                           const Approximator* const auxInputNet_) :
  settings(S), distrib(D), name(name_), replay(replay_),
  preprocessing(preprocessing_), auxInputNet(auxInputNet_)
{ }

void Approximator::setNumberOfAddedSamples(const Uint nSamples)
{
  if(bCreatedNetwork) die("cannot modify network setup after it was built");
  m_numberOfAddedSamples = nSamples;
}

//specify type (and size) of auxiliary input
void Approximator::setAddedInput(const ADDED_INPUT type, Sint size)
{
  if(bCreatedNetwork) die("cannot modify network setup after it was built");
  if(type == NONE)
  {
    if(size>0) die("No added input must have size 0");
    if(auxInputNet) die("Given auxInputNet Approximator but specified no added inputyo");
    m_auxInputSize = 0;
  }
  else if (type == NETWORK)
  {
    if(not auxInputNet) die("auxInputNet was not given on construction");
    if(size<0) m_auxInputSize = auxInputNet->nOutputs();
    else {
      m_auxInputSize = size;
      if(auxInputNet->nOutputs() < (Uint) size)
        die("Approximator allows inserting the first 'size' outputs of "
            "another 'auxInputNet' Approximator as additional input (along "
            "with the state or the output of 'preprocessing' Approximator). "
            "But auxInputNet's output must be at least of size 'size'.");
    }
  }
  else if (type == ACTION || type == VECTOR)
  {
    if(size<=0) die("Did not specify size of the action/vector");
    m_auxInputSize = size;
  } else die("type not recognized");
  if(m_auxInputSize<0) die("m_auxInputSize cannot be negative at this point");
}

// specify whether we are using target networks
void Approximator::setUseTargetNetworks(const Sint targetNetworkSampleID,
                                        const bool bTargetNetUsesTargetWeights)
{
  if(bCreatedNetwork) die("cannot modify network setup after it was built");
  m_UseTargetNetwork = true;
  m_bTargetNetUsesTargetWeights = bTargetNetUsesTargetWeights;
  m_targetNetworkSampleID = targetNetworkSampleID;
}

void Approximator::initializeNetwork()
{
  const MDPdescriptor & MDP = replay->MDP;
  if(build->layers.back()->bOutput == false)
  {
    warn("Requested net where last layer isnt output. Overridden: now it is");
    build->layers.back()->bOutput = true;
    build->nOutputs += build->layers.back()->size;
  }
  build->build();
  std::swap(net, build->net); std::swap(opt, build->opt);
  std::vector<std::shared_ptr<Parameters>> grads = build->threadGrads;
  assert(opt && net && grads.size() == nThreads);
  delete build.release();

  contexts.reserve(nThreads);
  #pragma omp parallel num_threads(nThreads)
  for (Uint i=0; i<nThreads; ++i)
  {
    if(i == (Uint) omp_get_thread_num())
      contexts.emplace_back(
        std::make_unique<ThreadContext>(i, grads[i],
                                        m_numberOfAddedSamples,
                                        m_UseTargetNetwork,
                                        m_bTargetNetUsesTargetWeights? -1 : 0));
    #pragma omp barrier
  }

  agentsContexts.reserve(nAgents);
  for (Uint i=0; i<nAgents; ++i)
    agentsContexts.emplace_back( std::make_unique<AgentContext>(i) );

  const auto& layers = net->layers;
  if (m_auxInputSize>0) // If we have an auxInput (eg policy for DPG) to what
  {                     // layer does it attach? Then we can grab gradient.
    auxInputAttachLayer = 0; // preprocessing/state and aux in one input layer
    for(Uint i=1; i<layers.size(); ++i) if(layers[i]->bInput) {
      if(auxInputAttachLayer>0) die("too many input layers, not supported");
      auxInputAttachLayer = i;
    }
    if (auxInputAttachLayer > 0) {
      if(layers[auxInputAttachLayer]->nOutputs() != auxInputNet->nOutputs())
        die("Size of layer to which auxInputNet does not match auxInputNet");
      if(preprocessing && layers[0]->nOutputs() != preprocessing->nOutputs())
        die("Mismatch in preprocessing output size and network input");
      const Uint stateInpSize = (1+MDP.nAppendedObs) * MDP.dimStateObserved;
      if(not preprocessing && layers[0]->nOutputs() != stateInpSize)
        die("Mismatch in state size and network input");
    }
    if(MDP.dimStateObserved > 0 and not layers[0]->bInput)
      die("Network does not have input layer.");
  }


  if (m_blockInpGrad or not preprocessing)
  {
    // Skip backprop to input vector or to preprocessing if 'm_blockInpGrad'
    // Three cases of interest:
    // 1) (most common) no aux input or both both preprocessing and aux input
    //    are given at layer 0 then block backprop at layer 1
    // 2) aux input given at layer greater than 1:  block backprop at layer 1
    // 3) aux input is layer 1 and layer 2 is joining (glue) layer, then
    //    gradient blocking is done at layer 3
    const Uint skipBackPropLayerID = auxInputAttachLayer==1? 3 : 1;
    if (auxInputAttachLayer==1) // check logic of statement 3)
      assert(layers[1]->bInput && not net->layers[2]->bInput);

    if (layers.size() > skipBackPropLayerID) {
      const Uint inputSize = preprocessing? preprocessing->nOutputs()
                           : (1+MDP.nAppendedObs) * MDP.dimStateObserved;
      if(auxInputAttachLayer==0) // check statement 1)
        assert(layers[1]->spanCompInpGrads == inputSize + m_auxInputSize);
      else if(auxInputAttachLayer==1) // check statement 3)
        assert(layers[3]->spanCompInpGrads == inputSize + m_auxInputSize);
      assert(layers[skipBackPropLayerID]->spanCompInpGrads >= inputSize);
      // next two lines actually tell the network to skip backprop to input:
      layers[skipBackPropLayerID]->spanCompInpGrads -= inputSize;
      layers[skipBackPropLayerID]->startCompInpGrads = inputSize;
    }
  }

  #ifdef __CHECK_DIFF //check gradients with finite differences
    net->checkGrads();
  #endif
  gradStats = new StatsTracker(net->getnOutputs(), distrib);
}

// buildFromSettings reads from the settings file the amount of fully connected
// layers (nnl1, nnl2, ...) and builds a network with given number of nInputs
// and nOutputs. Supports LSTM, RNN and MLP (aka InnerProduct or Dense).
//void stackSimple(Uint ninps,Uint nouts) { return stackSimple(ninps,{nouts}); }
void Approximator::buildFromSettings(const std::vector<Uint> outputSizes)
{
  if (not build)
    build = std::make_unique<Builder>(settings, distrib);

  const MDPdescriptor & MDP = replay->MDP;
  // last chance to update size of aux input size:
  if(auxInputNet && m_auxInputSize<=0) {
    assert(m_auxInputSize not_eq 0 && "Default is -1, what set it to 0?");
    m_auxInputSize = auxInputNet->nOutputs();
  }
  //build.stackSimple( inputSize, outputSizes );

  const Uint nOuts = std::accumulate(outputSizes.begin(), outputSizes.end(), 0);
  const std::string netType = settings.nnType, funcType = settings.nnFunc;
  const std::string outFuncType = settings.nnOutputFunc;
  const std::vector<Uint>& layerSizes = settings.nnLayerSizes;

  if( build->layers.size() )
  {
    if(preprocessing)
      die("Preprocessing layers were created for a network type that does not "
          "support being together with preprocessing layers");
    if(m_auxInputSize>0)
      build->addInput(m_auxInputSize); // add slot to insert aux input layer
  }
  else
  {
    Uint inputSize = preprocessing? preprocessing->nOutputs()
                                  : (1+MDP.nAppendedObs) * MDP.dimStateObserved;
    if(m_auxInputSize>0) inputSize += m_auxInputSize;

    if(inputSize == 0) {
      warn("network with no input space. will return a param layer");
      build->addParamLayer(nOuts, outFuncType, std::vector<Real>(nOuts, 0));
      return;
    } else
      build->addInput(inputSize);
  }

  //User can specify how many layers exist independendlty for each output
  // of the network. For example, if the settings file specifies 3 layer
  // sizes and splitLayers=1, the network will have 2 shared bottom Layers
  // (not counting input layer) and then for each of the outputs a separate
  // third layer each connected back to the second layer.
  const Uint nLayers = layerSizes.size();
  const Uint nsplit = std::min((Uint) settings.splitLayers, nLayers);
  const Uint firstSplit = nLayers - nsplit;

  for(Uint i=0; i<firstSplit; ++i)
    build->addLayer(layerSizes[i], funcType, false, netType);

  if(nOuts > 0)
  {
    if(nsplit)
    {
      const Uint lastShared = build->layers.size() - 1;
      for (Uint i=0; i<outputSizes.size(); ++i)
      {
        //`link' specifies how many layers back should layer take input from
        // use layers.size()-lastShared >=1 to link back to last shared layer
        const Uint nConnectBack = build->layers.size() - lastShared;
        build->addLayer(layerSizes[firstSplit], funcType, false,
                        netType, nConnectBack);

        for (Uint j=firstSplit+1; j<layerSizes.size(); ++j)
          build->addLayer(layerSizes[j], funcType, false, netType);

        build->addLayer(outputSizes[i], settings.nnOutputFunc, true);
      }
    }
    else build->addLayer(nOuts, settings.nnOutputFunc, true);
  }
}

void Approximator::buildPreprocessing(const std::vector<Uint> preprocLayers)
{
  if(build)
    die("attempted to create preprocessing layers multiple times");

  build = std::make_unique<Builder>(settings, distrib);

  const MDPdescriptor & MDP = replay->MDP;
  const Uint dimS = preprocessing? preprocessing->nOutputs()
                                 : (1+MDP.nAppendedObs) * MDP.dimStateObserved;
  if ( MDP.conv2dDescriptors.size() > 0 )
  {
    const Uint nConvs = MDP.conv2dDescriptors.size();
    const auto& conv0 = MDP.conv2dDescriptors[0];
    assert(dimS >= conv0.inpFeatures*conv0.inpY*conv0.inpX);
    const Sint extraInputSize = dimS - conv0.inpFeatures*conv0.inpY*conv0.inpX;

    build->addInput(conv0.inpFeatures * conv0.inpY * conv0.inpX );
    for(Uint i=0; i<nConvs; ++i)
      build->addConv2d(MDP.conv2dDescriptors[i]);

    if(extraInputSize) {
      warn("Mismatch between state dim and input conv2d, will add extra "
           "variables after the convolutional layers.");
      build->addInput(extraInputSize);
    }
  }

  for (Uint i=0; i<preprocLayers.size(); ++i)
    build->addLayer(preprocLayers[i], settings.nnFunc, false);
}

void Approximator::getHeaders(std::ostringstream& buff) const
{
  return opt->getHeaders(buff, name);
}
void Approximator::getMetrics(std::ostringstream& buff) const
{
  return opt->getMetrics(buff);
}

void Approximator::save(const std::string base, const bool bBackup)
{
  if(opt == nullptr) die("Attempted to save uninitialized net!");
  opt->save(base + name, bBackup);
}
void Approximator::restart(const std::string base)
{
  if(opt == nullptr) die("Attempted to restart uninitialized net!");
  opt->restart(base+name);
}

} // end namespace smarties


/*
Rvec forward(const Uint samp, const Uint thrID,
  const int USE_WGT, const int USE_ACT, const int overwrite=0) const;
inline Rvec forward(const Uint samp, const Uint thrID, int USE_ACT=0) const {
  assert(USE_ACT>=0);
  return forward(samp, thrID, thread_Wind[thrID], USE_ACT);
}
template<NET USE_A = CUR>
inline Rvec forward_cur(const Uint samp, const Uint thrID) const {
  const int indA = USE_A==CUR? 0 : -1;
  return forward(samp, thrID, thread_Wind[thrID], indA);
}
template<NET USE_A = TGT>
inline Rvec forward_tgt(const Uint samp, const Uint thrID) const {
  const int indA = USE_A==CUR? 0 : -1;
  return forward(samp, thrID, -1, indA);
}
// relay backprop requires gradients: no wID, no sorting based opt algos
Rvec relay_backprop(const Rvec grad, const Uint samp, const Uint thrID,
  const bool bUseTargetWeights = false) const;
void backward(Rvec grad, const Uint samp, const Uint thrID, const int USE_ACT=0) const;
void gradient(const Uint thrID, const int wID = 0) const;
void prepareUpdate();
void applyUpdate();
bool ready2ApplyUpdate();
*/
