//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Builder.h"

#include "CMA_Optimizer.h"
#include "Optimizer.h"
#include "Network.h"
#include "Layers/Layer_Base.h"
#include "Layers/Layer_Conv2D.h"
#include "Layers/Layer_LSTM.h"
#include "Layers/Layer_MGU.h"

using namespace smarties
{

Builder::Builder(const Settings& _sett) : settings(_sett) { }

void Builder::addInput(const int size)
{
  if(bBuilt) die("Cannot build the network multiple times");
  if(size<=0) die("Requested an empty input layer\n");
  const int ID = layers.size();
  layers.push_back(new InputLayer(size, ID));
  assert(layers[ID]->nOutputs() == (Uint) size);
  if(nInputs > 0) {
    const Uint twoLayersSize = layers[ID-1]->nOutputs() + size;
    layers.push_back(new JoinLayer(ID+1, twoLayersSize, 2));
  } else assert(ID == 0);
  nInputs += size;
}

void Builder::addLayer(const int nNeurons, const std::string funcType,
                       const bool bOutput, const std::string layerType,
                       const int iLink)
{
  if(bBuilt) die("Cannot build the network multiple times");
  const int ID = layers.size();
  if(iLink<1 || ID<iLink || layers[ID-iLink]==nullptr || nInputs==0)
    die("Missing input layer.");
  if(nNeurons <= 0)  die("Requested empty layer.");
  const Uint layInp = layers[ID-iLink]->nOutputs();
  Layer* l = nullptr;
         if (layerType == "LSTM") {
    l = new LSTMLayer(ID, layInp, nNeurons, funcType, bOutput, iLink);
  } else if (layerType == "MGU" || layerType == "GRU") {
    l = new MGULayer(ID, layInp, nNeurons, funcType, bOutput, iLink);
  } else if (layerType == "IntegrateFire") {
    //l = new IntegrateFireLayer(nInputs, nNeurons, layers.size());
  } else {
    const bool bRecur = (layerType=="RNN") || (layerType=="Recurrent");
    l = new BaseLayer(ID, layInp, nNeurons, funcType, bRecur, bOutput, iLink);
  }
  assert(l not_eq nullptr);
  layers.push_back(l);

  const bool bResLayer = (int) layers[ID-1]->nOutputs()==nNeurons && !bOutput;
  //const bool bResLayer = not bOutput;
  if(bResLayer)
    layers.push_back(new ResidualLayer(ID+1, nNeurons));

  if(bOutput) nOutputs += l->nOutputs();
}

void Builder::setLastLayersBias(std::vector<Real> init_vals)
{
  layers.back()->biasInitialValues(init_vals);
}

void Builder::addParamLayer(int size, std::string funcType, Real init_vals)
{
  addParamLayer(size, funcType, std::vector<Real>(size, init_vals) );
}

void Builder::addParamLayer(int size, std::string funcType, std::vector<Real> init_vals)
{
  const Uint ID = layers.size();
  if(bBuilt) die("Cannot build the network multiple times\n");
  if(size<=0) die("Requested an empty layer\n");
  Layer* l = new ParamLayer(ID, size, funcType, init_vals);
  layers.push_back(l);
  assert(l not_eq nullptr);
  nOutputs += l->nOutputs();
}

void Builder::stackSimple(const Uint ninps, const std::vector<Uint> nouts)
{
  const int sumout=static_cast<int>(accumulate(nouts.begin(),nouts.end(),0));
  const string netType = settings.nnType, funcType = settings.nnFunc;
  const vector<int> lsize = settings.readNetSettingsSize();

  if(ninps == 0)
  {
    warn("network with no input space. will return a param layer");
    addParamLayer(sumout, settings.nnOutputFunc, vector<Real>(sumout,0));
    return;
  }

  addInput(ninps);

  //User can specify how many layers exist independendlty for each output
  // of the network. For example, if the settings file specifies 3 layer
  // sizes and splitLayers=1, the network will have 2 shared bottom Layers
  // (not counting input layer) and then for each of the outputs a separate
  // third layer each connected back to the second layer.
  const Uint nL = lsize.size();
  const Uint nsplit = std::min((Uint) settings.splitLayers, nL);
  const Uint firstSplit = nL - nsplit;

  for(Uint i=0; i<firstSplit; i++) addLayer(lsize[i],funcType,false,netType);
  if(sumout) {
    if(nsplit) {
      const Uint lastShared = layers.back()->number();
      for (Uint i=0; i<nouts.size(); i++) {
        //`link' specifies how many layers back should layer take input from
        // use layers.size()-lastShared >=1 to link back to last shared layer
        addLayer(lsize[lastShared], funcType, false, netType, nL-lastShared);

        for (Uint j=firstSplit+1; j<lsize.size(); j++)
          addLayer(lsize[j], funcType, false, netType);

        addLayer(nouts[i], settings.nnOutputFunc, true);
      }
    } else addLayer(sumout, settings.nnOutputFunc, true);
  }
}

Network* Builder::build(const bool isInputNet)
{
  if(bBuilt) die("Cannot build the network multiple times\n");
  bBuilt = true;

  nLayers = layers.size();
  weights = allocate_parameters(layers, mpisize);
  tgt_weights = allocate_parameters(layers, mpisize);

  // Initialize weights
  for(const auto & l : layers)
    l->initialize(&generators[0], weights,
      l->bOutput && not isInputNet ? settings.outWeightsPrefac : 1);

  if(settings.learner_rank == 0) {
    for(const auto & l : layers) cout << l->printSpecs();
  }

  // Make sure that all ranks have the same weights (copy from rank 0)
  weights->broadcast(settings.mastersComm);
  //weights->allocateTransposed();
  tgt_weights->copy(weights); //copy weights onto tgt_weights

  // Allocate a gradient for each thread.
  Vgrad.resize(nThreads, nullptr);

  #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
  for (Uint i=0; i<Vgrad.size(); i++)
    #pragma omp critical // numa-aware allocation if OMP_PROC_BIND is TRUE
      Vgrad[i] = allocate_parameters(layers, mpisize);

  // Initialize network workspace to check that all is ok
  Activation*const test = allocate_activation(layers);

  if(test->nInputs not_eq (int) nInputs)
    _die("Mismatch between Builder's computed inputs:%u and Activation's:%u",
      nInputs, test->nInputs);

  if(test->nOutputs not_eq (int) nOutputs) {
    _warn("Mismatch between Builder's computed outputs:%u and Activation's:%u. Overruled Builder: probable cause is that user's net did not specify which layers are output. If multiple output layers expect trouble\n",
      nOutputs, test->nOutputs);
    nOutputs = test->nOutputs;
  }

  _dispose_object(test);

  popW = initWpop(weights, CMApopSize, mpisize);

  net = new Network(this, settings);
  if(CMApopSize>1) opt = new CMA_Optimizer(settings, weights,tgt_weights, popW);
  else opt = new AdamOptimizer(settings, weights,tgt_weights, popW, Vgrad);

  return net;
}

inline bool matchConv2D(Conv2D_Descriptor DESCR, int InX, int InY, int InC, int KnX, int KnY, int KnC, int Sx, int Sy, int Px, int Py, int OpX, int OpY)
{
  bool sameInp = DESCR.inpFeatures==InC && DESCR.inpY==InX && DESCR.inpX==InY;
  bool sameOut = DESCR.outFeatures==KnC && DESCR.outY==OpY && DESCR.outX==OpX;
  bool sameFilter  = DESCR.filterx==KnX && DESCR.filtery==KnY;
  bool sameStride  = DESCR.stridex== Sx && DESCR.stridey== Sy;
  bool samePadding = DESCR.paddinx== Px && DESCR.paddiny== Py;
  return sameInp && sameOut && sameFilter && sameStride && samePadding;
}

void Builder::addConv2d(Conv2D_Descriptor& descr, bool bOut, int iLink)
{
  if(bBuilt) die("Cannot build the network multiple times");
  const int ID = layers.size();
  if(iLink<1 || ID<iLink || layers[ID-iLink]==nullptr || nInputs==0)
    die("Missing input layer.");
  if( Kn_C*OutX*OutY <= 0 ) die("Requested empty layer.");
  if( layers[ID-iLink]->nOutputs() not_eq In_X * In_Y * In_C )
    _die("Mismatch between input size (%d) and previous layer size (%d).",
      In_X * In_Y * In_C, layers.back()->nOutputs() );

  Layer* l = nullptr;
  // I defined here the conv layers used in the Atari paper. To add new ones add
  // an if-pattern matching the other ones and refer to the `matchConv2D`
  // function above to interpret the arguments. Useful rule of thumb to remember
  // is: outSize should be : (InSize - FilterSize + 2*Padding)/Stride + 1
  if (      matchConv2D(descr, 84,84, 4, 8,8,32, 4,4, 0,0, 20,20) )
    l = new Conv2DLayer<LRelu, 84,84, 4, 8,8,32, 4,4, 0,0, 20,20>(ID,bOut,iL);
  else
  if (      matchConv2D(descr, 20,20,32, 4,4,64, 2,2, 0,0,  9, 9) )
    l = new Conv2DLayer<LRelu, 20,20,32, 4,4,64, 2,2, 0,0,  9, 9>(ID,bOut,iL);
  else
  if (      matchConv2D(descr,  9, 9,64, 3,3,64, 1,1, 0,0,  7, 7) )
    l = new Conv2DLayer<LRelu,  9, 9,64, 3,3,64, 1,1, 0,0,  7, 7>(ID,bOut,iL);
  else
    die("Detected undeclared conv2d description. This will be frustrating... "
        "In order to remove dependencies, keep the code low latency, and high "
        "performance, conv2d are templated. Whatever conv2d op you want must "
        "be specified in the Builder.cpp file. You'll see, it's easy.");

  layers.push_back(l);
  assert(l not_eq nullptr);
  if(bOutput) nOutputs += l->nOutputs();
}

} // end namespace smarties
