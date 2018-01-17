/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Parameters.h"
#include "Activation.h"
#include "Functions.h"
#include "../Profiler.h"

// Base class of all layer types. To insert a new layer type, overwrite all
// virtual functions.
class Layer
{
 public:
  const Uint size, ID, bInput;
  Uint bOutput;
  inline Uint number() const { return ID; }
  inline Uint nOutputs() const { return size; }

  // Should return the number of weights and biases required by layer
  virtual void requiredParameters(vector<Uint>& nWeight,
                                  vector<Uint>& nBiases ) const = 0;

  // Should return work memory that allows the network to compute forward step
  // and then, without re-calling forward, compute backward step.
  // See the LSTM class for an example on working out of the box.
  virtual void requiredActivation(vector<Uint>& sizes,
                                  vector<Uint>& bOutputs,
                                  vector<Uint>& bInputs) const = 0;
  // Some classes might allow user to specify an initial value for the bias
  // vector (eg. parametric layer or linear output layer)
  virtual void biasInitialValues(const vector<nnReal> init) = 0;

  Layer(Uint _ID, Uint _size, bool bOut, const bool bInp = false):
  size(_size), ID(_ID), bInput(bInp), bOutput(bOut)  {}
  virtual ~Layer() {}

  virtual void forward( const Activation*const prev,
                        const Activation*const curr,
                        const Parameters*const para) const = 0;
  // forward step without recurrent connection:
  inline void forward( const Activation*const curr,
                       const Parameters*const para) const {
    return forward(nullptr, curr, para);
  }

  virtual void backward( const Activation*const prev,
                         const Activation*const curr,
                         const Activation*const next,
                         const Parameters*const grad,
                         const Parameters*const para) const = 0;
  // forward step without recurrent connection:
  inline void backward( const Activation*const curr,
                        const Parameters*const grad,
                        const Parameters*const para) const {
    return backward(nullptr, curr, nullptr, grad, para);
  }

  // Initialize the weights and biases. Probably by sampling.
  virtual void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const = 0;
};

class InputLayer: public Layer
{
 public:
  InputLayer(Uint _size, Uint _ID) : Layer(_ID, _size, false, true) {
    printf("(%u) Input Layer of size:%u.\n", ID, size); fflush(0);
  }

  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    assert(nWeight.size() == 0 && nBiases.size() == 0);
    nWeight.push_back(0);
    nBiases.push_back(0);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    assert(sizes.size() == 0 && bOutputs.size() == 0);
    sizes.push_back(size);
    bOutputs.push_back(false);
    bOutputs.push_back(bInput);
  }
  void biasInitialValues(const vector<nnReal> init) override { }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override { }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override { }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override { }
};

class JoinLayer: public Layer
{
  const Uint nJoin;
 public:
  JoinLayer(Uint _ID, Uint _N, Uint _nJ): Layer(_ID,_N,false), nJoin(_nJ) {
    printf("(%u) Join Layer of size:%u.\n", ID, size); fflush(0);
    assert(nJoin>1);
  }

  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    assert(nWeight.size() == 0 && nBiases.size() == 0);
    nWeight.push_back(0);
    nBiases.push_back(0);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    assert(sizes.size() == 0 && bOutputs.size() == 0);
    sizes.push_back(size);
    bOutputs.push_back(bOutput);
    bOutputs.push_back(bInput);
  }
  void biasInitialValues(const vector<nnReal> init) override { }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override {
    nnReal* const ret = curr->Y(ID);
    Uint k = 0;
    for (Uint i=1; i<=nJoin; i++) {
      const nnReal* const inputs = curr->Y(ID-i);
      for (Uint j=0; j<curr->sizes[ID-i]; j++) ret[k++] = inputs[j];
    }
    assert(k==size);
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override {
    const nnReal* const errors = curr->E(ID);
    Uint k = 0;
    for (Uint i=1; i<=nJoin; i++) {
      nnReal* const ret = curr->E(ID-i);
      for (Uint j=0; j<curr->sizes[ID-i]; j++) ret[j] = errors[k++];
    }
    assert(k==size);
  }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override { }
};

class ParamLayer: public Layer
{
  const Function * const func;
  vector<nnReal> initVals;
 public:
  ~ParamLayer() { delete func; }
  ParamLayer(Uint _ID, Uint _size, string funcType, vector<nnReal> init) :
    Layer(_ID, _size, true), func(makeFunction(funcType)), initVals(init) {
    printf("(%u) %s ParameterLayer of size:%u.\n", ID, funcType.c_str(), size);
    if(initVals.size() != size) _die("size of initVals:%lu.", initVals.size());
    fflush(0);
  }

  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    nWeight.push_back(0); nBiases.push_back(size);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    sizes.push_back(size); bOutputs.push_back(true); bOutputs.push_back(bInput);
  }
  void biasInitialValues(const vector<nnReal> init) override {
    if(init.size() != size) _die("size of init:%lu.", init.size());
    initVals = init;
  }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
          nnReal* const inputs = curr->X(ID);
          nnReal* const output = curr->Y(ID);
    const nnReal* const bias = para->B(ID);
    for (Uint n=0; n<size; n++) {
      inputs[n] = bias[n];
      output[n] = func->eval(bias[n]);
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
          nnReal* const deltas = curr->E(ID);
          nnReal* const grad_b = grad->B(ID);
    const nnReal* const inputs = curr->X(ID);
    for(Uint o=0; o<size; o++) {
      deltas[o] *= func->evalDiff(inputs[o], deltas[o]);
      grad_b[o] += deltas[o];
    }
  }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override
  {
    nnReal* const biases = para->B(ID);
    for(Uint o=0; o<size; o++) biases[o] = func->inverse(initVals[o]);
  }
};


inline Activation* allocate_activation(const vector<Layer*>& layers) {
  vector<Uint> sizes, output, input;
  for(const auto & l : layers) l->requiredActivation(sizes, output, input);
  return new Activation(sizes, output, input);
}

inline Parameters* allocate_parameters(const vector<Layer*>& layers) {
  vector<Uint> nWeight, nBiases;
  for(const auto & l : layers) l->requiredParameters(nWeight, nBiases);
  return new Parameters(nWeight, nBiases);
}

inline Memory* allocate_memory(const vector<Layer*>& layers) {
  vector<Uint> sizes, output, input;
  for(const auto & l : layers) l->requiredActivation(sizes, output, input);
  return new Memory(sizes, output);
}
