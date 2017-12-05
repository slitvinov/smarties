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
#include "Link_Normal.h"

class IntegrateFireLayer: public BaseLayer<NormalLink>
{
  Sigm sigmoid;
 public:
  IntegrateFireLayer(Uint _nNeurons, Uint _iNeuron, Uint _iBias, const vector<NormalLink*> il, const Function*const _f, const Uint nn_simd, const bool bOut=true): BaseLayer(_nNeurons, _iNeuron, _iBias, il, nullptr, _f, nn_simd, bOut)
  {
    const string fname = "network_build.log";
    FILE * f = fopen(fname.c_str(), "a");
    if (f == NULL) die("Save fail\n");
    fprintf(f,"IntegrateFireLayer Layer of size %d, with first ID %d, and first bias ID %d\n", nNeurons, n1stNeuron, n1stBias);
    fflush(f);
    fclose(f);
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
