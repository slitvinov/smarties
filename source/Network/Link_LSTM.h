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

class LinkToLSTM : public Link
{
 public:
  /*
     if link is TO lstm, then the rules change a bit
     if a input signal is connected to one of the gates, is also connected to the others
     thus we just need the index of the first weight for the 3 gates (could have skipped this, iWi = iW + nO*nI and so forth)
     additionally the LSTM contains a memory, contained in Activation->ostate
     memory and gates are treated differently than normal neurons, therefore are contained in separate array, and i keep track of the position with iC
   */
  const Uint iC, iWI, iWF, iWO;

  LinkToLSTM(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iC, Uint _iW,
    Uint _iWI, Uint _iWF, Uint _iWO, Uint _nO_simd, Uint _nI_simd) :
    Link(_nI, _iI, _nO, _iO, _iW, _nO_simd, _nI_simd, _nI_simd*_nO_simd), iC(_iC),  iWI(_iWI), iWF(_iWF), iWO(_iWO) //i care nW per neuron, just for the asserts
  {
    assert(iW  % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iWI % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iWF % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iWO % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iI  % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iC  % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iO  % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(nO_simd % (VEC_WIDTH/sizeof(nnReal)) == 0);
    print();
    assert(iWI==iW +nW);
    assert(iWF==iWI+nW);
    assert(iWO==iWF+nW);
  }

  void print() const override
  {
    const string fname = "network_build.log";
    FILE * f = fopen(fname.c_str(), "a");
    if (f == NULL) die("Save fail\n");
    fprintf(f,"LSTM link: nInputs:%d IDinput:%d nOutputs:%d IDoutput:%d IDcell:%d IDweight:%d nWeights:%d nO_simd:%d nI_simd:%d\n",
      nI,iI,nO,iO,iC,iW,nW,nO_simd,nI_simd);
    fflush(f);
    fclose(f);
  }

  void initialize(mt19937*const gen, nnOpRet _weights,
        const Function*const func, const Real fac) const
  {
    const Real width = 2*std::max(nO,nI); //stupid workaround...
    const Real gateInit = std::sqrt(6./(width + nO));
    const Real funcInit = func->weightsInitFactor(width,nO)*fac;
    _initialize(gen, _weights, funcInit, iW,  nO, nI, nO_simd);
    _initialize(gen, _weights, gateInit, iWI, nO, nI, nO_simd);
    _initialize(gen, _weights, gateInit, iWF, nO, nI, nO_simd);
    _initialize(gen, _weights, gateInit, iWO, nO, nI, nO_simd);
  }

  void save(vector<nnReal> & out, nnOpRet _weights) const override
  {
    _save(out, _weights, iW,  nO, nI, nO_simd);
    _save(out, _weights, iWI, nO, nI, nO_simd);
    _save(out, _weights, iWF, nO, nI, nO_simd);
    _save(out, _weights, iWO, nO, nI, nO_simd);
  }

  void restart(vector<nnReal> & buf, nnOpRet _weights) const override
  {
    _restart(buf, _weights, iW,  nO, nI, nO_simd);
    _restart(buf, _weights, iWI, nO, nI, nO_simd);
    _restart(buf, _weights, iWF, nO, nI, nO_simd);
    _restart(buf, _weights, iWO, nO, nI, nO_simd);
  }

  inline void propagate(const Activation* const netFrom, Activation* const netTo, nnOpInp weights) const
  {
    nnOpInp inp = netFrom->outvals + iI; //outputs feeding into lstm layer
    nnOpRet inC = netTo->in_vals + iO; //input to cell's nonlin func
    nnOpRet inI = netTo->iIGates + iC; //input to input gate's nonlin func
    nnOpRet inF = netTo->iFGates + iC; //input to forget gate's nonlin func
    nnOpRet inO = netTo->iOGates + iC; //input to output gate's nonlin func

    for (Uint i = 0; i < nI; i++) {
      nnOpInp wC = weights + iW  + nO_simd*i; //weights that connect
      nnOpInp wI = weights + iWI + nO_simd*i; //from layers's output
      nnOpInp wF = weights + iWF + nO_simd*i; // to cell/gates input
      nnOpInp wO = weights + iWO + nO_simd*i;

      #pragma omp simd aligned(inp,inC,inI,inF,inO,wC,wI,wF,wO:VEC_WIDTH) \
        safelen(VEC_WIDTH)
      for (Uint o = 0; o < nO; o++) {
        inC[o] += inp[i] * wC[o];
        inI[o] += inp[i] * wI[o];
        inF[o] += inp[i] * wF[o];
        inO[o] += inp[i] * wO[o];
      }
    }
  }
  inline void sortWeights_bck_to_fwd(nnOpInp w_bck, nnOpRet w_fwd) const override
  {
    #pragma omp parallel for collapse(2)
    for (Uint i = 0; i < nI; i++)
    for (Uint o = 0; o < nO; o++) {
      w_fwd[iW  +nO_simd*i +o] = w_bck[iW  +nI_simd*o +i];
      w_fwd[iWI +nO_simd*i +o] = w_bck[iWI +nI_simd*o +i];
      w_fwd[iWF +nO_simd*i +o] = w_bck[iWF +nI_simd*o +i];
      w_fwd[iWO +nO_simd*i +o] = w_bck[iWO +nI_simd*o +i];
    }
  }
  inline void sortWeights_fwd_to_bck(nnOpInp w_fwd, nnOpRet w_bck) const override
  {
    #pragma omp parallel for collapse(2)
    for (Uint i = 0; i < nI; i++)
    for (Uint o = 0; o < nO; o++) {
      w_bck[iW  +nI_simd*o +i] = w_fwd[iW  +nO_simd*i +o];
      w_bck[iWI +nI_simd*o +i] = w_fwd[iWI +nO_simd*i +o];
      w_bck[iWF +nI_simd*o +i] = w_fwd[iWF +nO_simd*i +o];
      w_bck[iWO +nI_simd*o +i] = w_fwd[iWO +nO_simd*i +o];
    }
  }
  inline void backPropagate(Activation* const netFrom, const Activation* const netTo, nnOpInp weights, nnOpRet gradW) const
  {
    nnOpInp inp = netFrom->outvals + iI;
    nnOpRet err = netFrom->errvals + iI;
    nnOpInp dC = netTo->eMCell  +iC, dI = netTo->eIGates +iC;
    nnOpInp dF = netTo->eFGates +iC, dO = netTo->eOGates +iC;

    for (Uint o = 0; o < nO; o++) {
      nnOpInp wO = weights +iWO +nI_simd*o, wF = weights +iWF +nI_simd*o;
      nnOpInp wI = weights +iWI +nI_simd*o, wC = weights +iW  +nI_simd*o;
      nnOpRet gO = gradW +iWO +nI_simd*o, gF = gradW +iWF +nI_simd*o;
      nnOpRet gI = gradW +iWI +nI_simd*o, gC = gradW +iW  +nI_simd*o;

      #pragma omp simd aligned(inp, err, dC, dI, dF, dO, wC, wI, wF, wO, gC, \
       gI, gF, gO : VEC_WIDTH) safelen(VEC_WIDTH)
      for (Uint i = 0; i < nI; i++) {
        gC[i] += inp[i] * dC[o];
        gI[i] += inp[i] * dI[o];
        gF[i] += inp[i] * dF[o];
        gO[i] += inp[i] * dO[o];
        err[i]+= dO[o]*wO[i] + dC[o]*wC[i] + dI[o]*wI[i] + dF[o]*wF[i];
      }
    }
  }
};
