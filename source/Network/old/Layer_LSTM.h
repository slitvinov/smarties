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
#include "Link_LSTM.h"

class LSTMLayer: public BaseLayer<LinkToLSTM>
{
  const Uint n1stCell, n1stBiasIG, n1stBiasFG, n1stBiasOG;
  const Function* const gate;
  const Function* const cell;

 public:
  ~LSTMLayer() {_dispose_object(gate); _dispose_object(cell);}
  LSTMLayer(Uint _nNeurons, Uint _n1stNeuron, Uint _indState, Uint _n1stBias,
      Uint _n1stBiasIG, Uint _n1stBiasFG, Uint _n1stBiasOG,
      const vector<LinkToLSTM*> rl_il, const LinkToLSTM* const rl_rl,
      const Function*const _f, const Function*const _g,
      const Function*const _c, const Uint nn_simd, const bool bOut=false):
        BaseLayer(_nNeurons,_n1stNeuron,_n1stBias,rl_il,rl_rl,_f,nn_simd,bOut),
        n1stCell(_indState), n1stBiasIG(_n1stBiasIG), n1stBiasFG(_n1stBiasFG),
        n1stBiasOG(_n1stBiasOG), gate(_g), cell(_c)
  {
    const string fname = "network_build.log";
    FILE * f = fopen(fname.c_str(), "a");
    if (f == NULL) die("Save fail\n");
    fprintf(f,"LSTM Layer of size %d, with first ID %d, first cell ID %d, and first bias ID %d\n", nNeurons, n1stNeuron, n1stCell, n1stBias);
    fflush(f);
    fclose(f);
    assert(n1stBiasIG==n1stBias  +nn_simd);
    assert(n1stBiasFG==n1stBiasIG+nn_simd);
    assert(n1stBiasOG==n1stBiasFG+nn_simd);
  }

  void propagate(const Activation*const prev, Activation*const curr, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpRet outputI = curr->oIGates+n1stCell, outputF = curr->oFGates+n1stCell;
    nnOpRet outputO = curr->oOGates+n1stCell, outputC = curr->oMCell +n1stCell;
    nnOpRet inputs = curr->in_vals +n1stNeuron, inputI= curr->iIGates+n1stCell;
    nnOpRet inputF = curr->iFGates +n1stCell, inputO = curr->iOGates +n1stCell;
    nnOpInp oldState =(prev==nullptr? curr->ostates : prev->ostates) +n1stCell; //if nullptr then unused, but assigned for safety
    nnOpRet state = curr->ostates +n1stCell, output = curr->outvals +n1stNeuron;

    memcpy(inputs, biases+n1stBias  , nNeurons*sizeof(nnReal));
    memcpy(inputI, biases+n1stBiasIG, nNeurons*sizeof(nnReal));
    memcpy(inputF, biases+n1stBiasFG, nNeurons*sizeof(nnReal));
    memcpy(inputO, biases+n1stBiasOG, nNeurons*sizeof(nnReal));

    for (const auto & link : input_links)
      link->propagate(curr,curr,weights);

    if(recurrent_link not_eq nullptr && prev not_eq nullptr)
      recurrent_link->propagate(prev,curr,weights);

    func->eval(inputs,outputC,nNeurons_simd);
    gate->eval(inputI,outputI,nNeurons_simd);
    gate->eval(inputF,outputF,nNeurons_simd);
    gate->eval(inputO,outputO,nNeurons_simd);

    #pragma omp simd aligned(state, outputC, outputI, oldState, \
      outputF, output, outputO: VEC_WIDTH) safelen(simdWidth)
    for (Uint n=0; n<nNeurons; n++) {
      const Real oldStateUp = prev==nullptr? 0 : oldState[n]*outputF[n];
      state[n]=outputC[n]*outputI[n] + oldStateUp;
      output[n] = outputO[n] * state[n];
    }
  }

  void backPropagate(Activation*const prev, Activation*const curr, const Activation*const next, Grads* const grad, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpInp inputs = curr->in_vals +n1stNeuron,inputI= curr->iIGates +n1stCell;
    nnOpInp inputF = curr->iFGates +n1stCell, inputO = curr->iOGates +n1stCell;
    nnOpInp outputI = curr->oIGates +n1stCell;
    //nnOpInp outputF = curr->oFGates +n1stCell;
    nnOpInp outputO = curr->oOGates +n1stCell, outputC = curr->oMCell +n1stCell;
    nnOpRet deltas = curr->errvals +n1stNeuron;
    nnOpRet deltaI = curr->eIGates +n1stCell, deltaF = curr->eFGates +n1stCell;
    nnOpRet deltaO = curr->eOGates +n1stCell, deltaC = curr->eMCell +n1stCell;
    nnOpRet gradbiasC = grad->_B +n1stBias,   gradbiasI = grad->_B +n1stBiasIG;
    nnOpRet gradbiasF = grad->_B +n1stBiasFG, gradbiasO = grad->_B +n1stBiasOG;

    for (Uint n=0; n<nNeurons; n++)
    {
      const nnReal deltaOut = deltas[n];

      deltaC[n] = func->evalDiff(inputs[n], deltaC[n]) * outputI[n];
      deltaI[n] = gate->evalDiff(inputI[n], deltaI[n]) * outputC[n];
      deltaF[n] = (prev==nullptr) ? 0 : gate->evalDiff(inputF[n], deltaF[n]) * prev->ostates[n1stCell+n];
      deltaO[n] = gate->evalDiff(inputO[n], deltaO[n]) * deltaOut * curr->ostates[n1stCell+n];

      deltas[n] = deltaOut * outputO[n] + (next==nullptr ? 0 : next->errvals[n1stNeuron+n]*next->oFGates[n1stCell+n]);

      deltaC[n] *= deltas[n];
      deltaI[n] *= deltas[n];
      deltaF[n] *= deltas[n];
      //grad bias == delta:
      gradbiasC[n] += deltaC[n];
      gradbiasI[n] += deltaI[n];
      gradbiasF[n] += deltaF[n];
      gradbiasO[n] += deltaO[n];
    }

    for (const auto & link : input_links)
      link->backPropagate(curr,curr,weights,grad->_W);

    if(recurrent_link not_eq nullptr && prev not_eq nullptr)
      recurrent_link->backPropagate(prev,curr,weights,grad->_W);
  }

  void initialize(mt19937*const gen, nnOpRet weights, nnOpRet biases, Real initializationFac) const override
  {
    const Real biasesInit = ((initializationFac>0) ? initializationFac : 1) * func->biasesInitFactor(nNeurons); //usually 0
    uniform_real_distribution<nnReal> dis(-biasesInit, biasesInit);
    BaseLayer::initialize(gen, weights, biases, initializationFac);
    for (Uint w=n1stBiasIG; w<n1stBiasIG+nNeurons_simd; w++)
      biases[w] = dis(*gen) - LSTM_PRIME_FAC;
    for (Uint w=n1stBiasFG; w<n1stBiasFG+nNeurons_simd; w++)
      biases[w] = dis(*gen) + LSTM_PRIME_FAC;
    for (Uint w=n1stBiasOG; w<n1stBiasOG+nNeurons_simd; w++)
      biases[w] = dis(*gen) - LSTM_PRIME_FAC;
  }

  void save(std::vector<nnReal>& outWeights, std::vector<nnReal>& outBiases, nnOpRet _weights, nnOpRet _biases) const override
  {
    BaseLayer::save(outWeights, outBiases, _weights, _biases);
    for (Uint w=n1stBiasIG; w<n1stBiasIG+nNeurons; w++)
      outBiases.push_back(_biases[w]);
    for (Uint w=n1stBiasFG; w<n1stBiasFG+nNeurons; w++)
      outBiases.push_back(_biases[w]);
    for (Uint w=n1stBiasOG; w<n1stBiasOG+nNeurons; w++)
      outBiases.push_back(_biases[w]);
  }

  void restart(vector<nnReal>& bufWeights, vector<nnReal>& bufBiases,  nnOpRet _weights, nnOpRet _biases) const override
  {
    BaseLayer::restart(bufWeights, bufBiases, _weights, _biases);
    for(Uint w=0;w<nNeurons;w++) _biases[w+n1stBiasIG]=readCutStart(bufBiases);
    for(Uint w=0;w<nNeurons;w++) _biases[w+n1stBiasFG]=readCutStart(bufBiases);
    for(Uint w=0;w<nNeurons;w++) _biases[w+n1stBiasOG]=readCutStart(bufBiases);
  }

  void regularize(nnOpRet weights, nnOpRet biases, const Real lambda) const override
  {
    if(bOutput) return;
    BaseLayer::regularize(weights, biases, lambda);
  }
  void orthogonalize(nnOpRet _weights, nnOpInp _biases) const override {}
};
