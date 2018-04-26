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
#ifndef __STDC_VERSION__ //it should never be defined with g++
#define __STDC_VERSION__ 0
#endif
#if 1 //def NDEBUG
#include "cblas.h"
#endif

class BaseLayer: public Layer
{
  const Uint nInputs, nNeurons, bRecurrent, link, nInp_simd, nOut_simd;
  const Function* const func;
  vector<nnReal> initVals;

 public:
  void requiredParameters(vector<Uint>& nWeight,
                          vector<Uint>& nBiases ) const override {
    nWeight.push_back(nOut_simd * (bRecurrent? nInputs + nNeurons : nInputs));
    nBiases.push_back(nNeurons);
  }
  void requiredActivation(vector<Uint>& sizes,
                          vector<Uint>& bOutputs,
                          vector<Uint>& bInputs) const override {
    sizes.push_back(nNeurons);
    bOutputs.push_back(bOutput);
    bInputs.push_back(bInput);
  }
  void biasInitialValues(const vector<nnReal> init) override {
    if(init.size() != size) _die("size of init:%lu.", init.size());
    initVals = init;
  }
  ~BaseLayer() {
    _dispose_object(func);
  }

  BaseLayer(Uint _ID, Uint _nInputs, Uint _nNeurons, string funcType, bool bRnn,
    bool bOut, Uint iLink) : Layer(_ID, _nNeurons, bOut), nInputs(_nInputs),
    nNeurons(_nNeurons), bRecurrent(bRnn), link(iLink),
    nInp_simd(roundUpSimd(_nInputs)), nOut_simd(roundUpSimd(_nNeurons)),
    func(makeFunction(funcType)) {
      printf("(%u) %s %s%sInnerProduct Layer of size:%u linked to Layer:%u of size:%u.\n",
      ID, funcType.c_str(), bOutput? "output ":"", bRecurrent? "Recurrent-":"",
        nNeurons, ID-link, nInputs);
      fflush(0);
    }

  void forward( const Activation*const prev,
                const Activation*const curr,
                const Parameters*const para) const override
  {
    nnReal* const suminp = curr->X(ID); //array that contains W * Y_{-1} + B
    assert(para->NB(ID) == nNeurons);
    memcpy(suminp, para->B(ID), nNeurons*sizeof(nnReal));
    {
      const nnReal* const inputs = curr->Y(ID-link);
      const nnReal* const weight = para->W(ID);
      for (Uint i = 0; i < nInputs; i++)
      {
        const nnReal* const W = weight + nOut_simd*i;
        #pragma omp simd aligned(suminp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < nNeurons; o++)
          suminp[o] += inputs[i] * W[o];
      }
    }
    if(bRecurrent && prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
      const nnReal* const weight = para->W(ID) +nOut_simd*nInputs;
      for (Uint i = 0; i < nNeurons; i++)
      {
        const nnReal* const W = weight + nOut_simd*i;
        #pragma omp simd aligned(suminp, inputs, W : VEC_WIDTH)
        for (Uint o = 0; o < nNeurons; o++)
          suminp[o] += inputs[i] * W[o];
      }
    }
    func->eval(suminp, curr->Y(ID), nNeurons);
  }

  void backward(  const Activation*const prev,
                  const Activation*const curr,
                  const Activation*const next,
                  const Parameters*const grad,
                  const Parameters*const para) const override
  {
    nnReal* const deltas = curr->E(ID);
    {
            nnReal* const grad_b = grad->B(ID);
      const nnReal* const suminp = curr->X(ID);
      const nnReal* const outval = curr->Y(ID);
      for(Uint o=0; o<nNeurons; o++)
        deltas[o] *= func->evalDiff(suminp[o], outval[o]);

      //curr->clipDelta(ID, nLinkedTo);
      #pragma omp simd aligned(deltas, grad_b : VEC_WIDTH)
      for(Uint o=0; o<nNeurons; o++) grad_b[o] += deltas[o];
    }
    {
      const nnReal* const inputs = curr->Y(ID-link);
            nnReal* const grad_w = grad->W(ID);

      for(Uint i=0; i<nInputs;  i++) {
              nnReal* const G = grad_w + nOut_simd*i;
        #pragma omp simd aligned(deltas,inputs,G : VEC_WIDTH)
        for(Uint o=0; o<nNeurons; o++) G[o] += inputs[i] * deltas[o];
      }

      if( forceBackProp || not curr->input[ID-link] )
      {
              nnReal* const errors = curr->E(ID-link);
        const nnReal* const weight = para->W_T(ID);
        for(Uint o=0; o<nNeurons; o++) {
          const nnReal* const W = weight + nInp_simd*o;
          #pragma omp simd aligned(deltas,errors,W : VEC_WIDTH)
          for(Uint i=0; i<nInputs;  i++) errors[i] += W[i] * deltas[o];
        }
        //cblas_dgemv(CblasRowMajor, CblasNoTrans, nInputs, nNeurons, 1,
        //  weight, nOut_simd, deltas, 1, 1, errors, 1);
      }
    }
    if(bRecurrent && prev not_eq nullptr)
    {
      const nnReal* const inputs = prev->Y(ID);
            nnReal* const errors = prev->E(ID);
      const nnReal* const weight = para->W(ID) +nOut_simd*nInputs;
            nnReal* const grad_w = grad->W(ID) +nOut_simd*nInputs;

      for(Uint i=0; i<nNeurons;  i++) {
        nnReal* const G = grad_w + nOut_simd*i;
        #pragma omp simd aligned(deltas, inputs, G : VEC_WIDTH)
        for(Uint o=0; o<nNeurons; o++) G[o] += inputs[i] * deltas[o];
      }
      cblas_dgemv(CblasRowMajor, CblasNoTrans, nNeurons, nNeurons, 1,
        weight, nOut_simd, deltas, 1, 1, errors, 1);
    }
  }

  void initialize(mt19937* const gen, const Parameters*const para,
    Real initializationFac) const override
  {
    const nnReal fac = (initializationFac>0) ? initializationFac : 1;
    const nnReal init = fac * func->initFactor(nInputs, nNeurons);
    uniform_real_distribution<nnReal> dis(-init, init);
    {
      nnReal* const biases = para->B(ID);
      for(Uint o=0; o<nNeurons; o++)
        if(initVals.size() != nNeurons) biases[o] = dis(*gen);
        else biases[o] = func->inverse(initVals[o]);
    }
    {
      nnReal* const weight = para->W(ID);
      for(Uint i=0; i<nInputs;  i++) for(Uint o=0; o<nNeurons; o++)
        weight[o +nOut_simd*i] = dis(*gen);
    }
    if(bRecurrent)
    {
      nnReal* const weight = para->W(ID) +nOut_simd*nInputs;
      for(Uint i=0; i<nNeurons;  i++) for(Uint o=0; o<nNeurons; o++)
        weight[o +nOut_simd*i] = dis(*gen);
    }
  }

  void transpose(const Parameters*const para) const override
  {
    const nnReal* const W   = para->W(ID);
          nnReal* const W_T = para->W_T(ID);
    for(Uint i=0; i<nInputs;  i++)
      for(Uint o=0; o<nNeurons; o++) W_T[nInp_simd*o + i] = W[nOut_simd*i + o];
  }

  void orthogonalize(const Parameters*const para) const
  {
    nnReal* const weight = para->W(ID);
    nnReal* const biases = para->B(ID);
    for(Uint i=1; i<nNeurons; i++) {
      for(Uint j=0; j<i; j++) {
        nnReal u_d_u = 0.0, v_d_u = 0.0;
        for(Uint k=0; k<nInputs; k++) {
          u_d_u += weight[j +nNeurons*k] * weight[j +nNeurons*k];
          v_d_u += weight[j +nNeurons*k] * weight[i +nNeurons*k];
        }
        if( v_d_u < 0 ) continue;
        const nnReal overlap = nnSafeExp(-100*std::pow(biases[i]-biases[j],2));
        const nnReal fac = v_d_u/u_d_u * overlap;
        for(Uint k=0; k<nInputs; k++)
          weight[i +nNeurons*k] -= fac * weight[i +nNeurons*k];
      }
    }
  }
};
