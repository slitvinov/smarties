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

class Layer
{
 public:
  const Uint size, ID, bOutput;
  inline Uint number() const { return ID; }
  inline Uint nOutputs() const { return size; }
  virtual void requiredParameters(vector<Uint>& nWeight,
                                  vector<Uint>& nBiases ) const = 0;
  virtual void requiredActivation(vector<Uint>& sizes,
                                  vector<Uint>& bOutputs) const = 0;
  virtual void biasInitialValues(const vector<nnReal> init) = 0;
  Layer(Uint _ID, Uint _size, bool bOut): size(_size), ID(_ID), bOutput(bOut) {}
  virtual ~Layer() {}

  virtual void forward( const Activation*const prev,
                        const Activation*const curr,
                        const Parameters*const para) const = 0;

  inline void forward( const Activation*const curr,
                       const Parameters*const para) const {
    return forward(nullptr, curr, para);
  }

  virtual void backward( const Activation*const prev,
                         const Activation*const curr,
                         const Activation*const next,
                         const Parameters*const para,
                         const Parameters*const grad) const = 0;

  inline void backward( const Activation*const curr,
                        const Parameters*const para,
                        const Parameters*const grad) const {
    return backward(nullptr, curr, nullptr, para, grad);
  }

  virtual void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const = 0;
};

class InputLayer: public Layer
{
 public:
  InputLayer(Uint _size) : Layer(0, _size, false) {}

  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    nWeight.push_back(0);
    nBiases.push_back(0);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs) const override {
    sizes.push_back(size);
    bOutputs.push_back(false);
  }
  void biasInitialValues(const vector<nnReal> init) override { }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override { }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const para,
                  const Parameters*const grad) const override { }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override { }
};

class ParamLayer: public Layer
{
  const Function * const func;
  vector<nnReal> initVals;
 public:
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
                          vector<Uint>& bOutputs) const override {
    sizes.push_back(size); bOutputs.push_back(true);
  }
  void biasInitialValues(const vector<nnReal> init) override {
    if(init.size() != size) _die("size of init:%lu.", init.size());
    initVals = init;
  }
  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override {
    nnReal* const inputs = curr->X(ID); nnReal* const output = curr->Y(ID);
    const nnReal* const bias = para->B(ID);
    for (Uint n=0; n<size; n++) {
      inputs[n] = bias[n]; output[n] = func->eval(bias[n]);
    }
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const para,
                  const Parameters*const grad) const override {
          nnReal* const deltas = curr->E(ID);
          nnReal* const grad_b = grad->B(ID);
    const nnReal* const inputs = curr->X(ID);
    for(Uint o=0; o<size; o++) {
      deltas[o] *= func->evalDiff(inputs[o], deltas[o]);
      grad_b[o] += deltas[o];
    }
  }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override {
    nnReal* const biases = para->B(ID);
    for(Uint o=0; o<size; o++) biases[o] = func->inverse(initVals[o]);
  }
};
