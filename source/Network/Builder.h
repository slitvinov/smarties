/*
 *  LSTMNet.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Optimizer.h"
#include "Network.h"
#include "Layer_Base.h"
#include "Layer_Conv2D.h"
//#include "Layer_IntFire.h"
//#include "Layer_LSTM.h
#include <fstream>

inline Activation* allocate_activation(const vector<Layer*>& layers) {
  vector<Uint> sizes, output;
  for(const auto & l : layers) l->requiredParameters(sizes, output);
  return new Activation(sizes, output);
}

inline Parameters* allocate_parameters(const vector<Layer*>& layers) {
  vector<Uint> nWeight, nBiases;
  for(const auto & l : layers) l->requiredParameters(nWeight, nBiases);
  return new Parameters(nWeight, nBiases);
}

inline Memory* allocate_memory(const vector<Layer*>& layers) {
  vector<Uint> sizes, output;
  for(const auto & l : layers) l->requiredParameters(sizes, output);
  return new Memory(sizes, output);
}

class Builder
{
public:

  Network* build()
  {
    if(bBuilt) die("Cannot build the network multiple times\n");
    bBuilt = true;

    weights = allocate_parameters(layers);
    tgt_weights = allocate_parameters(layers);

    for(const auto & l : layers)
      l->initialize(&generators[0], weights, l->bOutput? settings.outWeightsPrefac : 1);

    weights->broadcast(settings.mastersComm);
    tgt_weights->copy(weights);

    Vgrad.resize(nThreads);
    #pragma omp parallel for
    for (Uint i=0; i<nThreads; i++)
      #pragma omp critical // numa-aware allocation if OMP_PROC_BIND is TRUE
        Vgrad[i] = allocate_parameters(layers);

    mem.resize(nAgents);
    for (Uint i=0; i<nAgents; ++i) mem[i] = allocate_memory(layers);

    timeSeries.resize(1);
    timeSeries[0] = allocate_activation(layers);

    if(timeSeries[0]->nInputs() not_eq nInputs)
      _die("Mismatch between Builder's computed inputs:%u and Activation's:%u",
        nInputs, timeSeries[0]->nInputs());

    if(timeSeries[0]->nOutputs not_eq nOutputs) {
      _warn("Mismatch between Builder's computed outputs:%u and Activation's:%u. Overruled Builder: probable cause is that user's net did not specify which layers are output. If multiple output layers expect trouble\n",
        nOutputs, timeSeries[0]->nOutputs);
      nOutputs = timeSeries[0]->nOutputs;
    }
    nLayers = layers.size();

    net = new Network(this, settings);
    #ifndef __EntropySGD
      opt = new Optimizer(settings, weights, tgt_weights);
    #else
      opt = new Optimizer(settings, weights, tgt_weights);
    #endif

    //if (!settings.learner_rank) opt->save("initial");
    //#ifndef NDEBUG
    //  MPI_Barrier(settings.mastersComm);
    //  opt->restart("initial");
    //  opt->save("restarted"+to_string(settings.learner_rank));
    //#endif
    return net;
  }

  void stackSimple(Uint ninps,Uint nouts) { return stackSimple(ninps,{nouts}); }
  void stackSimple(const Uint ninps, const vector<Uint> nouts)
  {
    const int sumout=static_cast<int>(accumulate(nouts.begin(),nouts.end(),0));
    const string netType = settings.nnType, funcType = settings.nnFunc;
    const vector<int> lsize = settings.readNetSettingsSize();
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

    if(nsplit) {
      const Uint lastShared = layers.back()->number();
      for (Uint i=0; i<nouts.size(); i++) {
        //`link' specifies how many layers back should layer take input from
        // we use layers.size()-lastShared >=1 to link back to last shared layer
        addLayer(lsize[lastShared], funcType, false, netType, nL - lastShared);

        for (Uint j=firstSplit+1; j<lsize.size(); j++)
          addLayer(lsize[j], funcType, false, netType);

        addLayer(nouts[i], "Linear", true);
      }
    } else addLayer(sumout, "Linear", true);
  }

private:
  bool bBuilt = false;
public:
  Uint nAgents, nThreads;
  Uint nInputs=0, nOutputs=0, nLayers=0;
  std::vector<std::mt19937>& generators;
  Parameters *weights, *tgt_weights;
  vector<Activation*> timeSeries;
  vector<Parameters*> Vgrad;
  vector<Layer*> layers;
  vector<Memory*> mem;
  Settings & settings;

  Network* net = nullptr;
  Optimizer* opt = nullptr;

  Builder(Settings& _sett): nAgents(_sett.nAgents), nThreads(_sett.nThreads),
    generators(_sett.generators), settings(_sett) {
    assert(nAgents>0 && nThreads>0);
  }

  void addInput(const int size)
  {
    if(bBuilt) die("Cannot build the network multiple times");
    if(nInputs>0 || layers.size()) die("More than one input layer?");
    if(size<=0) die("Requested an empty input layer\n");

    nInputs += size;
    layers.push_back(new InputLayer(size));
  }

  template<
  typename func,
  int In_X, int In_Y, int In_C, //input image: x:width, y:height, c:channels
  int Kn_X, int Kn_Y, int Kn_C,  //filter: x:width, y:height, c:channels
  int Sx=1, int Sy=1, //stride x/y
  int OutX=(In_X -Kn_X)/Sx+1,
  int OutY=(In_Y -Kn_Y)/Sy+1> //output image: same number of channels as KnC
  void addConv2d(const bool bOutput=false, const int iLink = 1)
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
    l = new ConvLayer<func, In_X,In_Y,In_C, Kn_X,Kn_Y,Kn_C, Sx,Sy, OutX,OutY>(
      ID, bOutput, iLink);

    layers.push_back(l);
    assert(l not_eq nullptr);
    if(bOutput) nOutputs += l->nOutputs();

    #if 0
      assert((OutX-1)*Sx +Kn_X >= In_X);
      assert((OutX-1)*Sx +Kn_X <  Kn_X+In_X);
      assert((OutY-1)*Sy +Kn_Y >= In_Y);
      assert((OutY-1)*Sy +Kn_Y <  Kn_Y+In_Y);
      if(Kn_X<=0 || Kn_Y<=0 || Kn_C<=0) die("Bad request for conv2D: filter");
      if(OutX<=0 || OutY<=0) die("Bad request for conv2D: outSize");
      if(Sx<0 || Sy<0) die("Bad request for conv2D: padding or stride\n");
      //assert(Kn_X >= Sx && Kn_Y >= Sy && PadX < Kn_X && PadY < Kn_Y);
    #endif
  }

  void addLayer(const int nNeurons, const string funcType,
    const bool bOutput=false, const string layerType="", const int iLink = 1)
  {
    if(bBuilt) die("Cannot build the network multiple times");
    const int ID = layers.size();
    if(iLink<1 || ID<iLink || layers[ID-iLink]==nullptr || nInputs==0)
      die("Missing input layer.");
    if(nNeurons <= 0)  die("Requested empty layer.");
    const Uint layInp = layers[ID-iLink]->nOutputs();

    Layer* l = nullptr;
           if (layerType == "LSTM") {
      //l = new LSTMLayer<func>(nInputs, nNeurons, layers.size());
    } else if (layerType == "IntegrateFire") {
      //l = new IntegrateFireLayer(nInputs, nNeurons, layers.size());
    } else {
      const bool bRecur = (layerType=="RNN") || (layerType=="Recurrent");
      l = new BaseLayer(ID, layInp, nNeurons, funcType, bRecur, bOutput, iLink);
    }

    layers.push_back(l);
    assert(l not_eq nullptr);
    if(bOutput) nOutputs += l->nOutputs();
  }

  void addParamLayer(int size, string funcType = "Linear", Real init_vals = 0)
  {
    addParamLayer(size, funcType, vector<Real>(size, init_vals) );
  }
  void addParamLayer(int size, string funcType, vector<Real> init_vals)
  {
    const Uint ID = layers.size();
    if(bBuilt) die("Cannot build the network multiple times\n");
    if(size<=0) die("Requested an empty layer\n");
    Layer* l = new ParamLayer(ID, size, funcType, init_vals);
    layers.push_back(l);
    assert(l not_eq nullptr);
    nOutputs += l->nOutputs();
  }
};
