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
#include "Link_Conv2D.h"

class Conv2DLayer : public BaseLayer<LinkToConv2D>
{
 public:
  Conv2DLayer(Uint _nNeurons, Uint _n1stNeuron, Uint _n1stBias, const vector<LinkToConv2D*> nl_il, const Function*const _f, const Uint nn_simd, const bool bOut = false) : BaseLayer(_nNeurons, _n1stNeuron, _n1stBias, nl_il, nullptr, _f, nn_simd, bOut)
  {
    const string fname = "network_build.log";
    FILE * f = fopen(fname.c_str(), "a");
    if (f == NULL) die("Save fail\n");
    fprintf(f,"Conv Layer of size %d, with first ID %d and first bias ID %d\n",
        nNeurons,n1stNeuron, n1stBias);
    fflush(f);
    fclose(f);
  }
  void regularize(nnOpRet weights, nnOpRet biases, const Real lambda) const override
  {
    if(bOutput) return;
    BaseLayer::regularize(weights, biases, lambda);
  }
  void orthogonalize(nnOpRet _weights, nnOpInp _biases) const override {}
};
