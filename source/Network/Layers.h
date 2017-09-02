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

  BaseLayer(const Uint _nNeurons, const Uint _n1stNeuron, const Uint _n1stBias, const vector<TLink*> nl_il, const TLink*const nl_rl, const Function*const f, const Uint nn_simd, const bool bOut) : Layer(_nNeurons, _n1stNeuron, _n1stBias, f, nn_simd, bOut), input_links(nl_il), recurrent_link(nl_rl) {}

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
    for (Uint n=0; n<nNeurons; n++) gradbias[n] += deltas[n];
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

class NormalLayer: public BaseLayer<NormalLink>
{
 public:
  NormalLayer(Uint _nNeurons, Uint _n1stNeuron, Uint _n1stBias,
      const vector<NormalLink*> nl_il, const NormalLink* const nl_rl,
      const Function*const f, const Uint nn_simd, const bool bOut = false) :
        BaseLayer(_nNeurons, _n1stNeuron, _n1stBias, nl_il, nl_rl, f, nn_simd, bOut)
  {
    printf("%s layer of size %d, with first ID %d and first bias ID %d\n",
        bOut?"Output":"Normal", nNeurons, n1stNeuron, n1stBias);
  }
  void regularize(nnOpRet weights, nnOpRet biases, const Real lambda) const override
  {
    if(bOutput) return;
    BaseLayer::regularize(weights, biases, lambda);
  }
  void orthogonalize(nnOpRet _weights, nnOpInp _biases) const override
  {
    if(bOutput) return;
    BaseLayer::orthogonalize(_weights, _biases);
  }
};

class Conv2DLayer : public BaseLayer<LinkToConv2D>
{
 public:
  Conv2DLayer(Uint _nNeurons, Uint _n1stNeuron, Uint _n1stBias, const vector<LinkToConv2D*> nl_il, const Function*const f, const Uint nn_simd, const bool bOut = false) : BaseLayer(_nNeurons, _n1stNeuron, _n1stBias, nl_il, nullptr, f, nn_simd, bOut)
  {
    printf("Conv2D Layer of size %d, with first ID %d and first bias ID %d\n",
        nNeurons,n1stNeuron, n1stBias);
  }
  void regularize(nnOpRet weights, nnOpRet biases, const Real lambda) const override
  {
    if(bOutput) return;
    BaseLayer::regularize(weights, biases, lambda);
  }
  void orthogonalize(nnOpRet _weights, nnOpInp _biases) const override {}
};

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
      const Function*const f, const Function*const g,
      const Function*const c, const Uint nn_simd, const bool bOut=false):
        BaseLayer(_nNeurons,_n1stNeuron,_n1stBias,rl_il,rl_rl,f,nn_simd,bOut),
        n1stCell(_indState), n1stBiasIG(_n1stBiasIG), n1stBiasFG(_n1stBiasFG),
        n1stBiasOG(_n1stBiasOG), gate(g), cell(c)
  {
    printf("LSTM Layer of size %d, with first ID %d, first cell ID %d, and first bias ID %d\n",
        nNeurons, n1stNeuron, n1stCell, n1stBias);
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
    nnOpInp biasC = biases +n1stBias,   biasI = biases +n1stBiasIG;
    nnOpInp biasF = biases +n1stBiasFG, biasO = biases +n1stBiasOG;
    nnOpInp oldState =(prev==nullptr? curr->ostates : prev->ostates) +n1stCell; //if nullptr then unused, but assigned for safety
    nnOpRet state = curr->ostates +n1stCell, output = curr->outvals +n1stNeuron;

    #pragma omp simd aligned(inputs, inputI, inputF, inputO, \
      biasC, biasI, biasF, biasO: VEC_WIDTH) safelen(simdWidth)
    for (Uint n=0; n<nNeurons; n++) {
      inputs[n] = biasC[n]; inputI[n] = biasI[n];
      inputF[n] = biasF[n]; inputO[n] = biasO[n];
    }

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

class IntegrateFireLayer: public BaseLayer<NormalLink>
{
  Sigm sigmoid;
 public:
  IntegrateFireLayer(Uint _nNeurons, Uint _iNeuron, Uint _iBias, const vector<NormalLink*> il, const Function*const f, const Uint nn_simd, const bool bOut=true): BaseLayer(_nNeurons, _iNeuron, _iBias, il, nullptr, f, nn_simd, bOut)
  {
    printf("IntegrateFireLayer Layer of size %d, with first ID %d, and first bias ID %d\n",
        nNeurons, n1stNeuron, n1stBias);
  }

  void propagate(const Activation*const prev, Activation*const curr, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpRet inputs = curr->in_vals +n1stNeuron;
    nnOpRet output = curr->outvals +n1stNeuron;

    //Biases array contains, for each output, the bias of the sigmoid,
    // the inverse time scale of the exp decay, and the coef of the sigmoid
    #ifndef INTEGRATEANDFIRESHARED
      nnOpInp bias   = biases +n1stBias +0*nNeurons;
      nnOpInp invTau = biases +n1stBias +1*nNeurons; //will pass through sigmoid
      nnOpInp excitr = biases +n1stBias +2*nNeurons;
      nnOpInp thresh = biases +n1stBias +3*nNeurons;
    #else //shared parameters
      nnOpInp bias   = biases +n1stBias +0;
      nnOpInp invTau = biases +n1stBias +1; //will pass through sigmoid
      nnOpInp excitr = biases +n1stBias +2;
      nnOpInp thresh = biases +n1stBias +3;
    #endif

    //prepare input to sigmoid: f(weights \cdot inputs + noise + bias )
    #ifndef INTEGRATEANDFIRESHARED
      for (Uint n=0; n<nNeurons; n++) inputs[n] = bias[n];
    #else //shared parameters
      for (Uint n=0; n<nNeurons; n++) inputs[n] = bias[0];
    #endif
    //add the dot product using normal fully connected layer
    for (const auto & link : input_links) link->propagate(curr,curr,weights);
    //evaluate sigmoid:
    sigmoid.eval(inputs, output, nNeurons_simd);

    //multiply by excitation coefficient
    #ifndef INTEGRATEANDFIRESHARED
      for (Uint n=0; n<nNeurons; n++) output[n] *= excitr[n];
    #else //shared parameters
      for (Uint n=0; n<nNeurons; n++) output[n] *= excitr[0];
    #endif

    //if not first of sequence, add the term depending on previous realization
    //note that we add threshold to output, so we need to subtract from prev
    if(prev not_eq nullptr)
      for (Uint n=0; n<nNeurons; n++) {
        #ifndef INTEGRATEANDFIRESHARED
          const nnReal oldOutp = prev->outvals[n1stNeuron+n] -thresh[n];
          output[n] += thresh[n] + oldOutp*(1-sigmoid.eval(invTau[n]));
        #else //shared parameters
          const nnReal oldOutp = prev->outvals[n1stNeuron+n] -thresh[0];
          output[n] += thresh[0] + oldOutp*(1-sigmoid.eval(invTau[0]));
        #endif
      }
  }

  void backPropagate(Activation*const prev, Activation*const curr, const Activation*const next, Grads* const grad, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpInp inputs = curr->in_vals +n1stNeuron;
    //nnOpInp output = curr->outvals +n1stNeuron;
    nnOpInp prevOut = prev!=nullptr ? prev->outvals +n1stNeuron : nullptr;
    nnOpRet dErr_curr = curr->errvals +n1stNeuron;

    //Biases array contains, for each output, the bias of the sigmoid,
    // the stdev of the noise inside the sigmoid, the inverse time scale
    // of the exp decay, and the coefficient of the sigmoid
    //nnOpInp bias   = biases +n1stBias +0*nNeurons;
    #ifndef INTEGRATEANDFIRESHARED
      //nnOpInp bias   = biases+n1stBias +0*nNeurons;
      nnOpInp invTau = biases+n1stBias +1*nNeurons;
      nnOpInp excitr = biases+n1stBias +2*nNeurons;
      nnOpInp thresh = biases+n1stBias +3*nNeurons;
      nnOpRet gradBias   = grad->_B+n1stBias +0*nNeurons;
      nnOpRet gradInvTau = grad->_B+n1stBias +1*nNeurons;
      nnOpRet gradExcitr = grad->_B+n1stBias +2*nNeurons;
      nnOpRet gradThresh = grad->_B+n1stBias +3*nNeurons;
    #else //shared parameters
      //nnOpInp bias   = biases+n1stBias +0;
      nnOpInp invTau = biases+n1stBias +1;
      nnOpInp excitr = biases+n1stBias +2;
      nnOpInp thresh = biases+n1stBias +3;
      nnOpRet gradBias   = grad->_B+n1stBias +0;
      nnOpRet gradInvTau = grad->_B+n1stBias +1;
      nnOpRet gradExcitr = grad->_B+n1stBias +2;
      nnOpRet gradThresh = grad->_B+n1stBias +3;
    #endif

    for (Uint n=0; n<nNeurons; n++) {
      // update the gradient to the parameters of the IaF neuron:
      #ifndef INTEGRATEANDFIRESHARED
        const nnReal boundedInvTau = sigmoid.eval(invTau[n]);
        const nnReal dBndInvTau = sigmoid.evalDiff(invTau[n], 0);
        if(prev!=nullptr) //again, remove threshold from integrated signal
          gradInvTau[n]+= -dErr_curr[n]*(prevOut[n]-thresh[n])*dBndInvTau;
        gradExcitr[n]  +=  dErr_curr[n]*sigmoid.eval(inputs[n]);
        gradThresh[n]  +=  dErr_curr[n];
      #else
        const nnReal boundedInvTau = sigmoid.eval(invTau[0]);
        const nnReal dBndInvTau = sigmoid.evalDiff(invTau[0], 0);
        if(prev!=nullptr)
          gradInvTau[0]+= -dErr_curr[n]*(prevOut[n]-thresh[0])*dBndInvTau;
        gradExcitr[0]  +=  dErr_curr[n]*sigmoid.eval(inputs[n]);
        gradThresh[0]  +=  dErr_curr[n];
      #endif
      //gradient of total error wrt to output of neuron depends on
      //current error, plus exponentially decaying dependence on future errors
      if(prev!=nullptr)
        prev->errvals[n+n1stNeuron] += (1-boundedInvTau)*dErr_curr[n];

      #ifndef INTEGRATEANDFIRESHARED
        //fully connected link expects to find in curr->errvals:
        // dErr_t / dInput_t, which is dErr_t / dOut_t (dot) dSigmoid / dInput_t
        // where Input_t is the input to the sigmoid (contained in inputs[n])
        dErr_curr[n] *= excitr[n]*sigmoid.evalDiff(inputs[n], 0);
        //grad of bias: dErr_t / dInput_t * dInput_t /dBias =  dErr_t / dInput_t
        gradBias[n]  += dErr_curr[n];
      #else
        dErr_curr[n] *= excitr[0]*sigmoid.evalDiff(inputs[n], 0);
        gradBias[0]  += dErr_curr[n];
      #endif
    }

    for(const auto& link: input_links)
      link->backPropagate(curr,curr,weights,grad->_W);
  }

  void initialize(mt19937*const gen, nnOpRet weights, nnOpRet biases, Real initializationFac) const override
  {
    BaseLayer::initialize(gen, weights, nullptr, initializationFac);
    #ifndef INTEGRATEANDFIRESHARED
      nnOpRet bias   = biases +n1stBias +0*nNeurons;
      nnOpRet invTau = biases +n1stBias +1*nNeurons;
      nnOpRet excitr = biases +n1stBias +2*nNeurons;
      nnOpRet thresh = biases +n1stBias +3*nNeurons;
      for(Uint w=0; w<nNeurons_simd; w++) bias[w]   = -1;
      for(Uint w=0; w<nNeurons_simd; w++) invTau[w] = .5;
      for(Uint w=0; w<nNeurons_simd; w++) excitr[w] = .5;
      for(Uint w=0; w<nNeurons_simd; w++) thresh[w] = -1;
    #else
      biases[n1stBias+0]= -1; biases[n1stBias+1]= .5;
      biases[n1stBias+2]= .5; biases[n1stBias+3]= -1;
    #endif
  }

  void save(std::vector<nnReal>& outWeights, std::vector<nnReal>& outBiases, nnOpRet _weights, nnOpRet _biases) const override
  {
    BaseLayer::save(outWeights, outBiases, _weights, nullptr);
    #ifndef INTEGRATEANDFIRESHARED
      nnOpInp bias   = _biases +n1stBias +0*nNeurons;
      nnOpInp invTau = _biases +n1stBias +1*nNeurons;
      nnOpInp excitr = _biases +n1stBias +2*nNeurons;
      nnOpInp thresh = _biases +n1stBias +3*nNeurons;
      for(Uint w=0; w<nNeurons; w++)  outBiases.push_back(bias[w]);
      for(Uint w=0; w<nNeurons; w++)  outBiases.push_back(invTau[w]);
      for(Uint w=0; w<nNeurons; w++)  outBiases.push_back(excitr[w]);
      for(Uint w=0; w<nNeurons; w++)  outBiases.push_back(thresh[w]);
    #else
      outBiases.push_back(_biases[n1stBias+0]);
      outBiases.push_back(_biases[n1stBias+1]);
      outBiases.push_back(_biases[n1stBias+2]);
      outBiases.push_back(_biases[n1stBias+3]);
    #endif
  }

  void restart(vector<nnReal>& bufWeights, vector<nnReal>& bufBiases,  nnOpRet _weights, nnOpRet _biases) const override
  {
    BaseLayer::restart(bufWeights, bufBiases, _weights, nullptr);
    #ifndef INTEGRATEANDFIRESHARED
      nnOpRet bias   = _biases +n1stBias +0*nNeurons;
      nnOpRet invTau = _biases +n1stBias +1*nNeurons;
      nnOpRet excitr = _biases +n1stBias +2*nNeurons;
      nnOpRet thresh = _biases +n1stBias +3*nNeurons;
      for(Uint w=0; w<nNeurons; w++) bias[w]   = readCutStart(bufBiases);
      for(Uint w=0; w<nNeurons; w++) invTau[w] = readCutStart(bufBiases);
      for(Uint w=0; w<nNeurons; w++) excitr[w] = readCutStart(bufBiases);
      for(Uint w=0; w<nNeurons; w++) thresh[w] = readCutStart(bufBiases);
    #else
      _biases[n1stBias+0] = readCutStart(bufBiases);
      _biases[n1stBias+1] = readCutStart(bufBiases);
      _biases[n1stBias+2] = readCutStart(bufBiases);
      _biases[n1stBias+3] = readCutStart(bufBiases);
    #endif
  }

  void regularize(nnOpRet weights, nnOpRet biases, const Real lambda) const override
  {
    BaseLayer::regularize(weights, nullptr, lambda);
  }
  void orthogonalize(nnOpRet _weights, nnOpInp _biases) const override {}
};

class ParamLayer: public BaseLayer<NormalLink>
{
 public:
  ParamLayer(Uint _nNeurons, Uint _iNeuron, Uint _iBias, const Function*const f, const Uint nn_simd, const bool bOut=true): BaseLayer(_nNeurons, _iNeuron, _iBias, std::vector<NormalLink*>(), nullptr, f, nn_simd, bOut)
  {
    printf("ParamLayer Layer of size %d, with first ID %d, and first bias ID %d\n", nNeurons, n1stNeuron, n1stBias);
  }

  void propagate(const Activation*const prev, Activation*const curr, nnOpInp weights, nnOpInp biases) const override
  {
    nnOpRet inputs = curr->in_vals +n1stNeuron;
    nnOpRet output = curr->outvals +n1stNeuron;
    nnOpInp bias   = biases +n1stBias;
    for (Uint n=0; n<nNeurons; n++) inputs[n] = bias[n];
    func->eval(inputs, output, nNeurons_simd);
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
