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
    //orthogonalize(n0, _weights, nOut, nAdded, n_simd);
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
  void orthogonalize(const Uint n0, nnOpRet _weights, Uint nOut, Uint nIn, Uint n_simd) const
  {
    if (nIn<nOut) return;

    for (Uint i=1; i<nOut; i++) {
      nnReal v_d_v_pre = 0.;
      for (Uint k=0; k<nIn; k++)
        v_d_v_pre += *(_weights+n0+k*n_simd+i)* *(_weights+n0+k*n_simd+i);
      if(v_d_v_pre<std::numeric_limits<nnReal>::epsilon())
        die("Initialization problem\n");

      for (Uint j=0; j<i;  j++) {
        nnReal u_d_u = 0.0;
        nnReal v_d_u = 0.0;
        for (Uint k=0; k<nIn; k++) {
          u_d_u += *(_weights+n0+k*n_simd+j)* *(_weights+n0+k*n_simd+j);
          v_d_u += *(_weights+n0+k*n_simd+j)* *(_weights+n0+k*n_simd+i);
        }
        if(u_d_u<std::numeric_limits<nnReal>::epsilon())
          die("Initialization problem\n");

        for (Uint k=0; k<nIn; k++)
          *(_weights+n0+k*n_simd+i) -= v_d_u/u_d_u * *(_weights+n0+k*n_simd+j);
      }

      nnReal v_d_v_post = 0.0;
      for (Uint k=0; k<nIn; k++)
        v_d_v_post += *(_weights+n0+k*n_simd+i)* *(_weights+n0+k*n_simd+i);

      if(v_d_v_post<std::numeric_limits<nnReal>::epsilon())
        die("Initialization problem\n");

      for (Uint k=0; k<nIn; k++)
        *(_weights+n0+k*n_simd+i) *= std::sqrt(v_d_v_pre/v_d_v_post);
    }
  }
  inline void regularize(nnOpRet weights, const Real lambda) const
  {
    //not sure:
    Lpenalization(weights, iW, nW, lambda);
  }
};

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
    cout<<"Normal link: nInputs="<<nI<<" IDinput="<<iI<<" nOutputs="<<nO
        <<" IDoutput"<<iO<<" IDweight"<<iW<<" nWeights"<<nW
        <<" nO_simd"<<nO_simd<<" nI_simd"<<nI_simd<<endl;
    fflush(0);
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
  inline void sortWeights_bck_to_fwd(nnOpInp w_bck, nnOpRet w_fwd) const override
  {
    #pragma omp parallel for collapse(2)
    for (Uint i = 0; i < nI; i++)
    for (Uint o = 0; o < nO; o++)
      w_fwd[iW +nO_simd*i +o] = w_bck[iW +nI_simd*o +i];
  }
  inline void sortWeights_fwd_to_bck(nnOpInp w_fwd, nnOpRet w_bck) const override
  {
    #pragma omp parallel for collapse(2)
    for (Uint i = 0; i < nI; i++)
    for (Uint o = 0; o < nO; o++)
      w_bck[iW +nI_simd*o +i] = w_fwd[iW +nO_simd*i +o];
  }
  inline void backPropagate(Activation*const netFrom, const Activation*const netTo, nnOpInp weights, nnOpRet gradW) const
  {
    nnOpInp inp = netFrom->outvals + iI;
    nnOpInp delta = netTo->errvals + iO;
    nnOpRet err = netFrom->errvals + iI;
    //inp = (nnOpInp)__builtin_assume_aligned(inp, VEC_WIDTH);
    //err = (nnOpRet)__builtin_assume_aligned(err, VEC_WIDTH);
    //delta = (nnOpInp)__builtin_assume_aligned(delta, VEC_WIDTH);
    #if 0
    for (Uint o = 0; o < nO; o++) {
      nnOpInp w = weights +iW +nI_simd*o;
      nnOpRet g = gradW +iW +nI_simd*o;
      //g = (nnOpRet)__builtin_assume_aligned(g, VEC_WIDTH);
      //w = (nnOpInp)__builtin_assume_aligned(w, VEC_WIDTH);
#pragma omp simd aligned(g,inp,delta,err,w : VEC_WIDTH) safelen(VEC_WIDTH)
      for (Uint i = 0; i < nI; i++) {
        g[i] += inp[i] * delta[o];
        err[i] += delta[o] * w[i];
      }
    }
    #else

    for (Uint o = 0; o < nO; o++) {
      nnOpInp w = weights +iW +nI_simd*o;
      #pragma omp simd aligned(delta,err,w : VEC_WIDTH) safelen(VEC_WIDTH)
      for (Uint i = 0; i < nI; i++) err[i] += delta[o] * w[i];
    }

    for (Uint o = 0; o < nO; o++) {
      nnOpRet g = gradW +iW +nI_simd*o;
      #pragma omp simd aligned(g,inp,delta : VEC_WIDTH) safelen(VEC_WIDTH)
      for (Uint i = 0; i < nI; i++) g[i] += inp[i] * delta[o];
    }

    #endif
  }
};

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
    cout << "LSTM link: nInputs="<< nI << " IDinput=" << iI
        << " nOutputs=" << nO << " IDoutput" << iO << " IDcell" << iC
        << " IDweight" << iW << " nWeights" << nW << " nO_simd"<<nO_simd << endl;
    fflush(0);
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
    nnOpInp inp = netFrom->outvals + iI;
    nnOpRet inC = netTo->in_vals + iO;
    nnOpRet inI = netTo->iIGates + iC;
    nnOpRet inF = netTo->iFGates + iC;
    nnOpRet inO = netTo->iOGates + iC;

    for (Uint i = 0; i < nI; i++) {
      nnOpInp wC = weights + iW  + nO_simd*i;
      nnOpInp wI = weights + iWI + nO_simd*i;
      nnOpInp wF = weights + iWF + nO_simd*i;
      nnOpInp wO = weights + iWO + nO_simd*i;

#pragma omp simd aligned(inp,inC,inI,inF,inO,wC,wI,wF,wO:VEC_WIDTH) safelen(VEC_WIDTH)
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
    nnOpInp dC = netTo->eMCell  +iC;
    nnOpInp dI = netTo->eIGates +iC;
    nnOpInp dF = netTo->eFGates +iC;
    nnOpInp dO = netTo->eOGates +iC;

    for (Uint o = 0; o < nO; o++) {
      nnOpInp wO = weights +iWO +nI_simd*o;
      nnOpInp wF = weights +iWF +nI_simd*o;
      nnOpInp wI = weights +iWI +nI_simd*o;
      nnOpInp wC = weights +iW  +nI_simd*o;
      nnOpRet gO = gradW +iWO +nI_simd*o;
      nnOpRet gF = gradW +iWF +nI_simd*o;
      nnOpRet gI = gradW +iWI +nI_simd*o;
      nnOpRet gC = gradW +iW  +nI_simd*o;

#pragma omp simd aligned(inp,err,dC,dI,dF,dO,wC,wI,wF,wO,gC,gI,gF,gO:VEC_WIDTH) safelen(VEC_WIDTH)
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

class LinkToConv2D : public Link
{
public:
  const Uint inputWidth, inputHeight, inputDepth;
  const Uint filterWidth, filterHeight, outputDepth_simd;
  const Uint outputWidth, outputHeight, outputDepth;
  const Uint strideX, strideY, padX, padY;

  LinkToConv2D(Uint _nI, Uint _iI, Uint _nO, Uint _iO, Uint _iW, Uint _nO_simd,
      Uint _inW, Uint _inH, Uint _inD, Uint _fW, Uint _fH, Uint _fN, Uint _outW,
      Uint _outH, Uint _sX=1, Uint _sY=1, Uint _pX=0, Uint _pY=0) :
        Link(_nI, _iI, _nO, _iO, _iW, _nO_simd, inputDepth, _fW*_fH*_nO_simd*_inD),
        inputWidth(_inW), inputHeight(_inH), inputDepth(_inD),
        filterWidth(_fW), filterHeight(_fH), outputDepth_simd(_nO_simd),
        outputWidth(_outW), outputHeight(_outH), outputDepth(_fN),
        strideX(_sX), strideY(_sY), padX(_pX), padY(_pY)
  {
    assert(inputDepth % (VEC_WIDTH/sizeof(nnReal)) == 0);
    //assert(outputDepth % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iW % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iI % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(iO % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(outputDepth_simd % (VEC_WIDTH/sizeof(nnReal)) == 0);
    assert(nW>0);
    assert(inputWidth*inputHeight*inputDepth == nI);
    assert(outputWidth*outputHeight*outputDepth == nO);
    //this class prescribes the bottom padding, let's figure out if the top one makes sense
    // inW_withPadding = inputWidth + bottomPad + topPad (where bottomPad = padX,padY)
    //first: All pixels of input are covered. topPad must be >=0, and stride leq than filter size
    assert((outputWidth -1)*strideX + filterWidth  >= inputWidth+padX);
    assert((outputHeight-1)*strideY + filterHeight >= inputHeight+padY);
    assert(filterWidth >= strideX && filterHeight >= strideY);
    //second condition: do not feed an output pixel only with padding
    assert((outputWidth -1)*strideX+filterWidth <filterWidth +inputWidth+padX);
    assert((outputHeight-1)*strideY+filterHeight<filterHeight+inputHeight+padY);
    assert(padX < filterWidth && padY < filterHeight);
    print();
  }
  void print() const override
  {
    printf("iW=%d, nI=%d, iI=%d, nO=%d, iO=%d, nW=%d\n",
        iW,nI,iI,nO,iO,nW);
    printf("inputWidth=%d, inputHeight=%d, inputDepth=%d\n",
        inputWidth, inputHeight, inputDepth);
    printf("outputWidth=%d, outputHeight=%d, outputDepth=%d (%d)\n",
        outputWidth, outputHeight, outputDepth, outputDepth_simd);
    printf("filterWidth=%d, filterHeight=%d, strideX=%d, strideY=%d, padX=%d, padY=%d\n",
        filterWidth, filterHeight, strideX, strideY, padX, padY);
    fflush(0);
  }
  void initialize(mt19937*const gen, nnOpRet _weights, const Function*const func, const Real fac) const
  {
    const Uint nAdded = filterWidth*filterHeight*inputDepth;
    assert(outputDepth_simd*nAdded == nW);
    const Real init = func->weightsInitFactor(nAdded, outputDepth)*fac;
    _initialize(gen, _weights, init, iW, outputDepth, nAdded, outputDepth_simd);
  }
  void save(vector<nnReal> & out, nnOpRet _weights) const override
  {
    const Uint nAdded = filterWidth*filterHeight*inputDepth;
    _save(out, _weights, iW, outputDepth, nAdded, outputDepth_simd);
  }
  void restart(vector<nnReal> & buf, nnOpRet _weights) const override
  {
    const Uint nAdded = filterWidth*filterHeight*inputDepth;
    _restart(buf, _weights, iW, outputDepth, nAdded, outputDepth_simd);
  }
  inline void propagate(const Activation* const netFrom, Activation* const netTo, nnOpInp weights) const
  {
    for(Uint ox=0; ox<outputWidth;  ox++)
    for(Uint oy=0; oy<outputHeight; oy++) {
      //starting position along input map for convolution with kernel:
      const int ix = static_cast<int>(ox*strideX) -  static_cast<int>(padX);
      const int iy = static_cast<int>(oy*strideY) -  static_cast<int>(padY);
      for(Uint fx=0; fx<filterWidth; fx++)
      for(Uint fy=0; fy<filterHeight; fy++) {
        //index along input map of the convolution op:
        const int cx=ix+static_cast<int>(fx);
        const int cy=iy+static_cast<int>(fy);
        //padding: skip addition if outside input boundaries
        if ( cx < 0 || cx >= static_cast<int>(inputWidth)
          || cy < 0 || cy >= static_cast<int>(inputHeight)) continue;

        nnOpInp inp = netFrom->outvals +iI +inputDepth*(cy +inputHeight*cx);
        nnOpRet out = netTo->in_vals +iO+outputDepth_simd*(oy+outputHeight*ox);

        for(Uint iz=0; iz<inputDepth; iz++) { //loop over inp feature maps:
          nnOpInp w = weights +iW +outputDepth_simd*(iz +inputDepth*(fy +filterHeight*fx));

#pragma omp simd aligned(out, inp, w : VEC_WIDTH) safelen(simdWidth)
          for(Uint fz=0; fz<outputDepth; fz++) //loop over number of kernels
            out[fz] += inp[iz] * w[fz];
        }
      }
    }
  }
  inline void sortWeights_bck_to_fwd(nnOpInp w_bck, nnOpRet w_fwd) const override
  {
    die("sortWeights_bck_to_fwd");
  }
  inline void sortWeights_fwd_to_bck(nnOpInp w_fwd, nnOpRet w_bck) const override
  {
    die("sortWeights_bck_to_fwd");
  }
  inline void backPropagate(Activation* const netFrom, const Activation* const netTo, nnOpInp weights, nnOpRet gradW) const
  {
    for(Uint ox=0; ox<outputWidth;  ox++)
    for(Uint oy=0; oy<outputHeight; oy++) {
      const int ix = static_cast<int>(ox*strideX) -  static_cast<int>(padX);
      const int iy = static_cast<int>(oy*strideY) -  static_cast<int>(padY);
      for(Uint fx=0; fx<filterWidth; fx++)
      for(Uint fy=0; fy<filterHeight; fy++) {
        const int cx=ix+static_cast<int>(fx), cy=iy+static_cast<int>(fy);
        //padding: skip addition if outside input boundaries
        if ( cx < 0 || static_cast<Uint>(cx) >= inputWidth
          || cy < 0 || static_cast<Uint>(cy) >= inputHeight) continue;

        nnOpInp inp = netFrom->outvals +iI +inputDepth*(cy +inputHeight*cx);
        nnOpRet err = netFrom->errvals +iI +inputDepth*(cy +inputHeight*cx);
        nnOpInp delta= netTo->errvals +iO+outputDepth_simd*(oy+outputHeight*ox);

        for(Uint iz=0; iz<inputDepth; iz++) {
          nnOpInp w = weights +iW +outputDepth_simd*(iz+ inputDepth*(fy+ filterHeight*fx));
          nnOpRet g = gradW +iW +outputDepth_simd*(iz +inputDepth*(fy +filterHeight*fx));

#pragma omp simd aligned(err, w, delta, g, inp : VEC_WIDTH) safelen(simdWidth)
          for(Uint fz=0; fz<outputDepth; fz++) {
            err[iz] += w[fz]*delta[fz];
            g[fz] += inp[iz]*delta[fz];
          }
        }
      }
    }
  }
  #if 0
  static inline Uint encode(const Uint j, const Uint i, const Uint stride)
  {
    return i + j*stride; //c encoding
  }
  inline void enConv(const Activation* const netFrom, Activation* const netTo, nnOpInp weights) const
  {
    const int kernel_row = filterWidth * filterHeight * inputDepth;
    const int kernel_col = outputWidth * outputHeight;
    nnReal*const inp = initClean(kernel_row * kernel_col); //padding
    Uint o = 0; // A output size x kernelsize
    for(Uint ox=0; ox<outputWidth;  ox++)
    for(Uint oy=0; oy<outputHeight; oy++) {
      //starting position along input map for convolution with kernel:
      const int ix = ox*strideX - padX, iy = oy*strideY - padY;
      Uint f = 0;
      for(Uint fx=0; fx<filterWidth; fx++)
      for(Uint fy=0; fy<filterHeight; fy++) {
        //index along input map of the convolution op:
        const int cx=ix+fx, cy=iy+fy;
        //padding: skip addition if outside input boundaries
        if ( cx < 0 || cx >= inputWidth || cy < 0 || cy >= inputHeight) {
          f++;
          continue;
        }

        for(Uint iz=0; iz<inputDepth; iz++) {
          const Uint i = encode(o, f++, kernel_size);
          inp[i] = netFrom->outvals[iI +iz +inputDepth*(cy +inputHeight*cx)];
        }
      }
      o++;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
      kernel_col, outputDepth, kernel_row,
      1.0, inp, kernel_row, weights +iW, outputDepth,
      1.0, biases +n1stBias, outputDepth);
    _myfree(inp);
  }
  nnOpInp inp = netFrom->outvals +iI +inputDepth*(cy +inputHeight*cx);
  nnOpRet out = netTo->in_vals +iO+outputDepth_simd*(oy+outputHeight*ox);

  for(Uint iz=0; iz<inputDepth; iz++) //loop over inp feature maps:
    nnOpInp w=weights+iW+outputDepth_simd*(iz+inputDepth*(fy +filterHeight*fx));
    for(Uint fz=0; fz<outputDepth; fz++) out[fz] += inp[iz] * w[fz];

  #endif
};
