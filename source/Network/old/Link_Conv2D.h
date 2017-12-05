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
    const string fname = "network_build.log";
    FILE * f = fopen(fname.c_str(), "a");
    if (f == NULL) die("Save fail\n");

    fprintf(f,"iW=%d, nI=%d, iI=%d, nO=%d, iO=%d, nW=%d\n",
        iW,nI,iI,nO,iO,nW);
    fprintf(f,"inputWidth=%d, inputHeight=%d, inputDepth=%d\n",
        inputWidth, inputHeight, inputDepth);
    fprintf(f,"outputWidth=%d, outputHeight=%d, outputDepth=%d (%d)\n",
        outputWidth, outputHeight, outputDepth, outputDepth_simd);
    fprintf(f,"filterWidth=%d, filterHeight=%d, strideX=%d, strideY=%d, padX=%d, padY=%d\n",
        filterWidth, filterHeight, strideX, strideY, padX, padY);
    fflush(f);
    fclose(f);
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
          nnOpRet g = gradW +iW +outputDepth_simd*(iz +inputDepth*(fy +filterHeight*fx));
          #pragma omp simd aligned(delta,g,inp:VEC_WIDTH) safelen(simdWidth)
          for(Uint fz=0; fz<outputDepth; fz++)  g[fz] += inp[iz]*delta[fz];
        }
        for(Uint iz=0; iz<inputDepth; iz++) {
          nnOpInp w = weights +iW +outputDepth_simd*(iz+ inputDepth*(fy+ filterHeight*fx));
          #pragma omp simd aligned(err,w,delta:VEC_WIDTH) safelen(simdWidth)
          for(Uint fz=0; fz<outputDepth; fz++)  err[iz] += w[fz]*delta[fz];
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
