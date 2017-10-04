/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Activations.h"
#include <iostream>

class Link
{
 public:
  const Uint iW, nI, iI, nO, iO, nO_simd, nI_simd, nW;
  Link(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iW, Uint _nO_simd, Uint _nI_simd, Uint _nW) : iW(_iW), nI(_nI), iI(_iI), nO(_nO), iO(_iO), nO_simd(_nO_simd), nI_simd(_nI_simd), nW(_nW) {}

  virtual ~Link() {}
  virtual void print() const = 0;
  virtual inline void sortWeights_bck_to_fwd(nnOpInp w_bck, nnOpRet w_fwd) const
  {
    die("Only normal links\n");
  }
  virtual inline void sortWeights_fwd_to_bck(nnOpInp w_fwd, nnOpRet w_bck) const
  {
    die("Only normal links\n");
  }
  void _initialize(mt19937* const gen, nnOpRet _weights, const Real scale,
      Uint n0, Uint nOut, Uint nIn, Uint n_simd) const
  {
    assert(scale>0);
    uniform_real_distribution<nnReal> dis(-scale,scale);
    //normal_distribution<Real> dis(0.,range);
    for (Uint i = 0; i < nIn; i++)
      for (Uint o = 0; o < nOut; o++)
        _weights[n0 + n_simd*i + o] = dis(*gen);
  }
  virtual void save(vector<nnReal>& out, nnOpRet _weights) const = 0;
  void _save(vector<nnReal>& out, nnOpRet _weights, Uint n0, Uint nOut, Uint nIn, Uint n_simd) const
  {
    for (Uint i = 0; i < nIn; i++)
    for (Uint o = 0; o < nOut; o++) {
      const Uint w = n0 + n_simd*i + o;
      out.push_back(_weights[w]);
      assert(!std::isnan(_weights[w]) && !std::isinf(_weights[w]));
    }
  }
  virtual void restart(vector<nnReal>& buf, nnOpRet _weights) const = 0;
  void _restart(vector<nnReal>& buf, nnOpRet _weights, Uint n0, Uint nOut, Uint nIn, Uint n_simd) const
  {
    for (Uint i = 0; i < nIn; i++)
    for (Uint o = 0; o < nOut; o++) {
      const Uint w = n0 + n_simd*i + o;
      _weights[w] = buf.front();
      buf.erase(buf.begin(),buf.begin()+1);
      assert(!std::isnan(_weights[w]) && !std::isinf(_weights[w]));
    }
  }
  virtual void orthogonalize(nnOpRet _weights, nnOpInp _biases, const Uint firstBias) const {}
  inline void regularize(nnOpRet weights, const Real lambda) const
  {
    //not sure:
    Lpenalization(weights, iW, nW, lambda);
  }
};
