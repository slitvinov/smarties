/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Links.h"
#include "../Profiler.h"

class Layer
{
 public:
  const Uint nNeurons, n1stNeuron, n1stBias, nNeurons_simd;
  const Function* const func;
  //Profiler* profiler;
  const bool bOutput;

  virtual ~Layer() { _dispose_object(func); }
  Layer(const Uint _nNeurons, const Uint _n1stNeuron, const Uint _n1stBias,
      const Function*const f, const Uint nn_simd, const bool bOut) :
        nNeurons(_nNeurons), n1stNeuron(_n1stNeuron), n1stBias(_n1stBias),
        nNeurons_simd(nn_simd), func(f), bOutput(bOut) {}

  virtual void propagate(const Activation*const prev, Activation*const curr, nnOpInp weights, nnOpInp biases) const = 0;

  virtual void backPropagate(Activation*const prev, Activation*const curr, const Activation*const next, Grads*const grad, nnOpInp weights, nnOpInp biases) const = 0;

  virtual void initialize(mt19937* const gen, nnOpRet weights, nnOpRet biases, Real initializationFac) const = 0;

  virtual void save(vector<nnReal>& outWeights, vector<nnReal>& outBiases, nnOpRet _weights, nnOpRet _biases) const = 0;

  virtual void restart(vector<nnReal>& bufWeights, vector<nnReal>& bufBiases, nnOpRet _weights, nnOpRet _biases) const = 0;

  virtual void regularize(nnOpRet weights, nnOpRet biases, const Real lambda) const = 0;
  virtual void orthogonalize(nnOpRet weights, nnOpInp biases) const=0;

  void propagate(Activation*const curr, nnOpInp weights, nnOpInp biases) const
  {
    return propagate(nullptr, curr, weights, biases);
  }
  void backPropagate(Activation*const curr, Grads*const grad,  nnOpInp weights, nnOpInp biases) const
  {
    return backPropagate(nullptr, curr, nullptr, grad, weights, biases);
  }
};
