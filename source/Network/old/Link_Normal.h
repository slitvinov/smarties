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
#include "Graph.h"
#include <iostream>

class NormalLink: public Link
{
 public:
  /*
     a link here is defined as link layer to layer:
     index iI along the network activation outvals representing the index of the first neuron of input layer
     the number nI of neurons of the input layer
     the index iO of the first neuron of the output layer
     the number of neurons in the output layer nO
     the index of the first weight iW along the weight vector
     the weights are all to all: so this link occupies space iW to (iW + nI*nO) along weight vector
   */
  NormalLink(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iW, Uint _nO_simd, Uint _nI_simd) :
    Link(_nI, _iI, _nO, _iO, _iW, _nO_simd, _nI_simd, _nI_simd*_nO_simd)
  {
    assert(iW % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iI % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iO % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(nO_simd % (VEC_WIDTH/sizeof(nnReal)) == 0);
    print();
    assert(nI>0 && nO>0);
  }

  void print() const
  {
    const string fname = "network_build.log";
    FILE * f = fopen(fname.c_str(), "a");
    if (f == NULL) die("Save fail\n");
    fprintf(f,"Normal link: nInputs:%d IDinput:%d nOutputs:%d IDoutput:%d IDweight:%d nWeights:%d nO_simd:%d nI_simd:%d\n",
      nI,iI,nO,iO,iW,nW,nO_simd,nI_simd);
    fflush(f); fclose(f);
  }

  void save(vector<nnReal> & out, nnOpRet _weights) const override
  {
    _save(out, _weights, iW, nO, nI, nO_simd);
  }

  void restart(vector<nnReal> & buf, nnOpRet _weights) const override
  {
    _restart(buf, _weights, iW, nO, nI, nO_simd);
  }

  void initialize(mt19937*const gen, nnOpRet _weights, const Function*const func, const Real fac) const
  {
    const Real init = func->weightsInitFactor(nI, nO)*fac;
    _initialize(gen, _weights, init, iW, nO, nI, nO_simd);
  }

  //Links are from specific layer to specific layer:
  // propagate inp_i = sum_i,j w_i,j out_j ( Layer.h does out_i = f(inp_i) )
  // netFrom is network state in which out_j of input layer are stored
  // netTo is network state in which inp_i of output layer are to be computed
  inline void propagate(const Activation*const netFrom, Activation*const netTo, nnOpInp weights) const
  {
    nnOpInp inp = netFrom->outvals +iI;
    nnOpRet out = netTo->in_vals +iO;

    for (Uint i = 0; i < nI; i++) {
      nnOpInp w = weights +iW +nO_simd*i;
      #pragma omp simd aligned(inp,out,w : VEC_WIDTH) safelen(VEC_WIDTH)
      for (Uint o = 0; o < nO; o++) out[o] += inp[i] * w[o];
    }
  }
};
