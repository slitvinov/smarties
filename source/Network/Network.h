/*
 *  LSTMNet.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layer_Base.h"
#include "Layer_Conv2D.h"
#include "Layer_IntFire.h"
#include "Layer_LSTM.h"
#include "Layer_Normal.h"
#include "Layer_Param.h"

class Builder;

class Network
{
protected:
  const Uint nAgents, nThreads, nInputs, nOutputs, nLayers;
  const Uint nNeurons, nWeights, nBiases, nStates;
  const bool bDump;
public:
  const vector<Layer*> layers;
  const vector<Link*> links;
  nnReal* const weights;
  nnReal* const weights_back;
  nnReal* const biases;
  nnReal* const tgt_weights_back;
  nnReal* const tgt_weights;
  nnReal* const tgt_biases;
  Grads* const grad;
  const vector<Grads*> Vgrad;
  const vector<Mem*> mem;
  vector<std::mt19937>& generators;
  const vector<Uint> iOut, iInp;
  vector<Uint> dump_ID;
  const bool allocatedFrozenWeights = true;

  Uint getnWeights() const {return nWeights;}
  Uint getnBiases() const {return nBiases;}
  Uint getnOutputs() const {return nOutputs;}
  Uint getnInputs() const {return nInputs;}
  Uint getnNeurons() const {return nNeurons;}
  Uint getnStates() const {return nStates;}
  Uint getnLayers() const {return nLayers;}
  Uint getnAgents() const {return nAgents;}
  inline void sortWeights_bck_to_fwd() const
  {
    for (auto & l : links) l->sortWeights_bck_to_fwd(weights_back,weights);
  }
  inline void sortWeights_fwd_to_bck() const
  {
    for (auto & l : links) l->sortWeights_fwd_to_bck(weights,weights_back);
  }
  inline void sort_bck_to_fwd(nnReal*const _bck, nnReal*const _fwd) const
  {
    for (auto & l : links) l->sortWeights_bck_to_fwd(_bck, _fwd);
  }
  inline void sort_fwd_to_bck(nnReal*const _fwd, nnReal*const _bck) const
  {
    for (auto & l : links) l->sortWeights_fwd_to_bck(_fwd, _bck);
  }

  inline vector<Real> getOutputs(const Activation* const act) const
  {
    vector<Real> _output(nOutputs);
    for(Uint i=0; i<nOutputs; i++) _output[i] = act->outvals[iOut[i]];
    return _output;
  }
  inline vector<Real> getInputGradient(const Activation* const act) const
  {
    vector<Real> ret(nInputs);
    for(Uint j=0; j<nInputs; j++) ret[j]= act->errvals[iInp[j]];
    return ret;
  }
  inline void appendUnrolledActivations(vector<Activation*>* const ret, const Uint length) const
  {
    for(Uint j=0; j<length; j++)
      ret->push_back(new Activation(nNeurons,nStates));
  }
  inline Activation* allocateActivation() const
  {
    return new Activation(nNeurons,nStates);
  }
  inline vector<Activation*> allocateUnrolledActivations(const Uint len) const
  {
    vector<Activation*> ret(len);
    for (Uint j=0; j<len; j++) ret[j] = new Activation(nNeurons,nStates);
    return ret;
  }
  inline void prepForBackProp(vector<Activation*>* series,const Uint len) const
  {
    vector<Activation*>& ref = *series;
    if (series->size() < len)
      for(Uint j=series->size(); j<len; j++)
        series->push_back(new Activation(nNeurons,nStates));

    for(Uint j=0; j<len; j++) ref[j]->clearErrors();
  }
  inline void prepForFwdProp(vector<Activation*>* series, Uint len) const
  {
    if (series->size() < len)
      for(Uint j=series->size(); j<len; j++)
        series->push_back(new Activation(nNeurons,nStates));
  }
  static inline void deallocateUnrolledActivations(vector<Activation*>*const r)
  {
    for (auto & trash : *r) _dispose_object(trash);
    r->clear();
  }
  inline void clearErrors(vector<Activation*>& timeSeries) const
  {
    for (Uint k=0; k<timeSeries.size(); k++) timeSeries[k]->clearErrors();
  }
  inline void setOutputDeltas(const vector<Real>&_err, Activation*const a) const
  {
    assert(_err.size()==nOutputs);
    for (Uint i=0; i<nOutputs; i++) a->errvals[iOut[i]] = _err[i];
  }

  Network(Builder* const B, Settings & settings) ;

  ~Network()
  {
    for (auto & trash : layers) _dispose_object(trash);
    for (auto & trash : mem) _dispose_object(trash);
    for (auto & trash : Vgrad) _dispose_object(trash);
    _dispose_object( grad);
    _myfree( weights );
    _myfree( biases );
    _myfree( tgt_weights );
    _myfree( tgt_biases );
  }

  void updateFrozenWeights();

  void seqPredict_inputs(const vector<Real>& _input, Activation* const currActivation) const;
  void seqPredict_output(vector<Real>&_output, Activation* const currActivation) const;
  void seqPredict_execute(const vector<Activation*>& series_1, vector<Activation*>& series_2, const Uint start, const nnReal* const _weights, const nnReal* const _biases) const;
  inline void seqPredict_execute(const vector<Activation*>& series_1, vector<Activation*>& series_2, const nnReal* const _weights, const nnReal* const _biases) const
  {
    seqPredict_execute(series_1, series_2, 0, _weights, _biases);
  }
  inline void seqPredict_execute(const vector<Activation*>& series_1, vector<Activation*>& series_2, const Uint start) const
  {
    seqPredict_execute(series_1, series_2, start, weights, biases);
  }
  inline void seqPredict_execute(const vector<Activation*>& series_1, vector<Activation*>& series_2) const
  {
    seqPredict_execute(series_1, series_2, weights, biases);
  }

  void predict(const vector<Real>& _input, vector<Real>& _output,
      vector<Activation*>& timeSeries, const Uint n_step,
      const nnReal* const _weights, const nnReal* const _biases) const;
  inline void predict(const vector<Real>& _input, vector<Real>& _output,
      vector<Activation*>& timeSeries, const Uint n_step) const
  {
    predict(_input, _output, timeSeries, n_step, weights, biases);
  }

  void predict(const vector<Real>& _input, vector<Real>& _output,
   const Activation* const prevActivation, Activation* const currActivation,
   const nnReal* const _weights, const nnReal* const _biases) const;
  inline void predict(const vector<Real>& _input, vector<Real>& _output,
   const Activation*const prevActivation, Activation*const currActivation) const
  {
    predict(_input, _output, prevActivation, currActivation,
        weights, biases);
  }

  void predict(const vector<Real>& _input, vector<Real>& _output,
      Activation* const net, const nnReal* const _weights,
      const nnReal* const _biases) const;
  inline void predict(const vector<Real>& _input, vector<Real>& _output,
      Activation* const net) const
  {
    predict(_input, _output, net, weights, biases);
  }

  void backProp(vector<Activation*>& timeSeries,
      const nnReal* const _weights, const nnReal* const _biases,
      Grads* const _grads) const;
  inline void backProp(vector<Activation*>& timeSeries, Grads* const _grads) const
  {
    backProp(timeSeries, weights_back, biases, _grads);
  }

  void backProp(vector<Activation*>& timeSeries, const Uint len, const nnReal* const _weights, const nnReal* const _biases, Grads* const _grads) const;
  inline void backProp(vector<Activation*>& timeSeries, const Uint len, Grads* const _grads) const
  {
    backProp(timeSeries, len, weights_back, biases, _grads);
  }

  void backProp(const vector<Real>& _errors, Activation* const net,
      const nnReal* const _weights, const nnReal* const _biases,
      Grads* const _grads) const;
  inline void backProp(const vector<Real>& _errors, Activation* const net,
      Grads* const _grads) const
  {
    backProp(_errors, net, weights_back, biases, _grads);
  }

  void checkGrads();
  inline void regularize(const Real lambda) const
  {
    #pragma omp parallel for
    for (Uint j=0; j<nLayers; j++)
      layers[j]->regularize(weights_back, biases, lambda);
  }
  inline void orthogonalize() const
  {
    #pragma omp parallel for
    for (Uint j=0; j<nLayers; j++)
      layers[j]->orthogonalize(weights_back, biases);
  }

  void save(vector<nnReal> & outWeights, vector<nnReal> & outBiases,
      nnReal* const _weights, nnReal* const _biases) const
  {
    for (const auto &l : layers)
      l->save(outWeights,outBiases, _weights, _biases);
  }
  void restart(vector<nnReal> & outWeights, vector<nnReal> & outBiases,
      nnReal* const _weights, nnReal* const _biases) const
  {
    for (const auto &l : layers)
      l->restart(outWeights,outBiases, _weights, _biases);
  }
  //void save(const string fname);
  void dump(const int agentID);
  //bool restart(const string fname);
};
