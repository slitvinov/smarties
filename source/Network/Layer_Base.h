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

template<typename TLink>
class BaseLayer: public Layer
{
 public:
  const vector<TLink*> input_links;
  const TLink* const recurrent_link;

  virtual ~BaseLayer()
  {
    for (auto & trash : input_links) _dispose_object(trash);
    _dispose_object(recurrent_link);
  }

  BaseLayer(const Uint _nNeurons, const Uint _n1stNeuron, const Uint _n1stBias, const vector<TLink*> nl_il, const TLink*const nl_rl, const Function*const _f, const Uint nn_simd, const bool bOut) : Layer(_nNeurons, _n1stNeuron, _n1stBias, _f, nn_simd, bOut), input_links(nl_il), recurrent_link(nl_rl) {}

  virtual void propagate(const Activation*const prev, Activation*const curr, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpRet outputs = curr->outvals +n1stNeuron;
    nnOpRet inputs = curr->in_vals +n1stNeuron;
    nnOpInp bias = biases +n1stBias;
    //const int thrID = omp_get_thread_num();
    //if(thrID==1) profiler->push_start("FB");
    #pragma omp simd aligned(inputs,bias : VEC_WIDTH) safelen(simdWidth)
    for (Uint n=0; n<nNeurons; n++) inputs[n] = bias[n];
    //if(thrID==1)  profiler->stop_start("FP");
    for (const auto & link : input_links)
      link->propagate(curr,curr,weights);
    if(recurrent_link not_eq nullptr && prev not_eq nullptr)
      recurrent_link->propagate(prev,curr,weights);

    //if(thrID==1)  profiler->stop_start("FD");
    //for (Uint n=0; n<nNeurons; n++) outputs[n] = func->eval(inputs[n]);
    func->eval(inputs,outputs,nNeurons_simd);
    //if(thrID==1) profiler->pop_stop();
  }

  virtual void backPropagate(Activation*const prev, Activation*const curr, const Activation*const next, Grads*const grad, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpInp inputs = curr->in_vals +n1stNeuron;
    nnOpRet deltas = curr->errvals +n1stNeuron;
    nnOpRet gradbias = grad->_B +n1stBias;
    //const int thrID = omp_get_thread_num();
    //if(thrID==1) profiler->push_start("BD");
    for(Uint n=0;n<nNeurons;n++) deltas[n]*=func->evalDiff(inputs[n],deltas[n]);

    //if(thrID==1)  profiler->stop_start("BP");
    for (const auto & link : input_links)
      link->backPropagate(curr,curr,weights,grad->_W);

    if(recurrent_link not_eq nullptr && prev not_eq nullptr)
      recurrent_link->backPropagate(prev,curr,weights,grad->_W);

    //if(thrID==1)  profiler->stop_start("BB");
    #pragma omp simd aligned(gradbias,deltas: VEC_WIDTH) safelen(simdWidth)
    for (Uint n=0; n<nNeurons; n++) gradbias[n] += deltas[n] -1e-6*inputs[n];
    //if(thrID==1) profiler->pop_stop();
  }

  virtual void initialize(mt19937* const gen, nnOpRet weights, nnOpRet biases, Real initializationFac) const override
  {
    const nnReal prefac = (initializationFac>0) ? initializationFac : 1;
    const nnReal biasesInit =prefac*func->biasesInitFactor(nNeurons);//usually 0
    uniform_real_distribution<nnReal> dis(-biasesInit, biasesInit);

    if(biases not_eq nullptr)
    for (Uint w=n1stBias; w<n1stBias+nNeurons_simd; w++)
      biases[w] = dis(*gen);

    for (const auto & link : input_links)
      if(link not_eq nullptr) link->initialize(gen,weights,func,prefac);

    if(recurrent_link not_eq nullptr)
      recurrent_link->initialize(gen,weights,func,prefac);
  }

  virtual void save(vector<nnReal>& outWeights, vector<nnReal>& outBiases, nnOpRet _weights, nnOpRet _biases) const override
  {
    for (const auto & l : input_links)
      if(l not_eq nullptr) l->save(outWeights, _weights);

    if(recurrent_link not_eq nullptr)
      recurrent_link->save(outWeights, _weights);

    if(_biases not_eq nullptr)
    for (Uint w=n1stBias; w<n1stBias+nNeurons; w++)
      outBiases.push_back(_biases[w]);
  }

  virtual void restart(vector<nnReal>& bufWeights, vector<nnReal>& bufBiases, nnOpRet _weights, nnOpRet _biases) const override
  {
    for (const auto & l : input_links)
      if(l not_eq nullptr) l->restart(bufWeights, _weights);

    if(recurrent_link not_eq nullptr)
      recurrent_link->restart(bufWeights, _weights);

    if(_biases not_eq nullptr)
    for(Uint w=0; w<nNeurons; w++) _biases[w+n1stBias]=readCutStart(bufBiases);
  }

  virtual void regularize(nnOpRet weights, nnOpRet biases, const Real lambda) const override
  {
    for (const auto & link : input_links)
      link->regularize(weights, lambda);

    if(recurrent_link not_eq nullptr)
      recurrent_link->regularize(weights, lambda);

    if(biases not_eq nullptr)
    Lpenalization(biases, n1stBias, nNeurons, lambda);
  }
  virtual void orthogonalize(nnOpRet weights, nnOpInp bias) const override
  {
    for(const auto &link:input_links)link->orthogonalize(weights,bias,n1stBias);
  }
};
