/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"

class ParamLayer: public BaseLayer<NormalLink>
{
 public:
  ParamLayer(Uint _nNeurons, Uint _iNeuron, Uint _iBias, const Function*const _f, const Uint nn_simd, const bool bOut=true): BaseLayer(_nNeurons, _iNeuron, _iBias, std::vector<NormalLink*>(), nullptr, _f, nn_simd, bOut)
  {
    const string fname = "network_build.log";
    FILE * f = fopen(fname.c_str(), "a");
    if (f == NULL) die("Save fail\n");
    fprintf(f,"ParamLayer Layer of size %d, with first ID %d, and first bias ID %d\n", nNeurons, n1stNeuron, n1stBias);
    fflush(f);
    fclose(f);
  }

  void propagate(const Activation*const prev, Activation*const curr, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpRet inputs = curr->in_vals +n1stNeuron;
    nnOpRet output = curr->outvals +n1stNeuron;
    nnOpInp bias   = biases +n1stBias;
    for (Uint n=0; n<nNeurons; n++) inputs[n] = bias[n];
    for (Uint n=0; n<nNeurons; n++) output[n] = func->eval(inputs[n]);
    //printf("ParamLayer: %u %f %f %f\n", n1stNeuron, inputs[0], bias[0], output[0]); fflush(0);
  }

  void backPropagate(Activation*const prev, Activation*const curr, const Activation*const next, Grads* const grad, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpInp inputs = curr->in_vals +n1stNeuron;
    nnOpRet dErr_curr = curr->errvals +n1stNeuron;
    nnOpRet gradBias = grad->_B +n1stBias;
    for (Uint n=0; n<nNeurons; n++)  {
      dErr_curr[n] *= func->evalDiff(inputs[n],dErr_curr[n]);
      gradBias[n] += dErr_curr[n];
    }
  }

  void initialize(mt19937*const gen, nnOpRet weights, nnOpRet biases, Real initializationFac) const override
  {
    for(Uint w=0; w<nNeurons_simd; w++) biases[w+n1stBias] = initializationFac;
  }
  void save(std::vector<nnReal>& outWeights, std::vector<nnReal>& outBiases, nnOpRet _weights, nnOpRet _biases) const override
  {
    for(Uint w=0; w<nNeurons; w++) outBiases.push_back(_biases[w+n1stBias]);
  }
  void restart(vector<nnReal>& bufWeights, vector<nnReal>& bufBiases,  nnOpRet _weights, nnOpRet _biases) const override
  {
    for(Uint w=0; w<nNeurons; w++) _biases[w+n1stBias]=readCutStart(bufBiases);
  }
  void regularize(nnOpRet weights, nnOpRet biases, const Real lambda) const override { }
  void orthogonalize(nnOpRet _weights, nnOpInp _biases) const override {}
};
