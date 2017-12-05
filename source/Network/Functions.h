/*
 *  Activations.h
 *  rl
 *
 *  Guido Novati on 04.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <cmath>
#include "../Settings.h"
#include "Utils.h"
using namespace std;
#ifndef PRELU_FAC
#define PRELU_FAC 0.01
#endif

typedef nnReal* __restrict__       const nnOpRet;
typedef const nnReal* __restrict__ const nnOpInp;
//List of non-linearities for neural networks
//- eval return f(in), also present as array in / array out
//- evalDiff returns f'(x)
//- initFactor: some prefer fan in fan out, some only fan-in dependency
//If adding a new function, edit function readFunction at end of file

struct Function {
  //weights are initialized with uniform distrib [-weightsInitFactor, weightsInitFactor]
  virtual Real initFactor(const Uint inps, const Uint outs) const = 0;

  virtual void eval(nnOpInp in, nnOpRet out, const Uint N) const = 0; // f(in)

  virtual nnReal eval(const nnReal in) const = 0;
  virtual nnReal inverse(const nnReal in) const = 0; // f(in)
  virtual nnReal evalDiff(const nnReal in, const nnReal d) const = 0; // f'(in)

  virtual ~Function() {}
};

struct Linear : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    memcpy(out, in, N*sizeof(nnReal));
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    memcpy(out, in, N*sizeof(nnReal));
  }
  static inline nnReal _eval(const nnReal in) {
    return in;
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    return 1;
  }
  nnReal eval(const nnReal in) const override {
    return in;
  }
  nnReal inverse(const nnReal in) const override {
    return in;
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return 1;
  }
};

struct Tanh : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(6./(inps + outs));
  }
  static inline nnReal _eval(const nnReal in) {
    if(in >   EXP_CUT) return  1;
    if(in < - EXP_CUT) return -1;
    if(in > 0) {
      const nnReal e2x = std::exp(-2*in);
      return (1-e2x)/(1+e2x);
    } else {
      const nnReal e2x = std::exp( 2*in);
      return (e2x-1)/(1+e2x);
    }
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    const nnReal arg = in < 0? -in : in; //symmetric
    const nnReal e2x = std::exp(-2*arg);
    //if (arg > EXP_CUT && d*in > 0) return 0;
    if (arg > EXP_CUT) return 4*e2x;
    return 4*e2x/((1+e2x)*(1+e2x));
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(std::fabs(in)<1);
    return 0.5 * std::log((1+in)/(1-in));
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

struct Sigm : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(6./(inps + outs));
  }
  static inline nnReal _eval(const nnReal in) {
    if(in >  2*EXP_CUT) return 1;
    if(in < -2*EXP_CUT) return 0;
    if(in > 0) return 1/(1+std::exp(-in));
    else {
      const nnReal ex = std::exp(in);
      return ex/(1+ex);
    }
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    const nnReal arg = in < 0 ? -in : in;
    const nnReal ex = std::exp(-arg);
    if (arg > 2*EXP_CUT) return ex;
    return ex/((1+ex)*(1+ex));
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in > 0 && in < 1);
    return - std::log(1/in - 1);
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

struct HardSign : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(6./(inps + outs));
  }
  static inline nnReal _eval(const nnReal in) {
    return in/std::sqrt(1+in*in);
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    const nnReal denom = std::sqrt(1+in*in);
    return 1/(denom*denom*denom);
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(VEC_WIDTH)
    for (Uint i=0; i<N; i++) out[i] = in[i]/std::sqrt(1+in[i]*in[i]);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in > 0 && in < 1);
    return in/std::sqrt(1 -in*in);
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

struct SoftSign : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(6./(inps + outs));
  }
  static inline nnReal _eval(const nnReal in) {
    return in/(1+std::fabs(in));
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    const nnReal denom = 1+std::fabs(in);
    return 1/(denom*denom);
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(VEC_WIDTH)
    for (Uint i=0;i<N; i++) out[i] = in[i]/(1+std::fabs(in[i]));
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in > 0 && in < 1);
    return in/(1-in);
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

struct Relu : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return in>0 ? in : 0;
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    return in>0 ? 1 : 0;
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(VEC_WIDTH)
    for (Uint i=0;i<N; i++) out[i] = in[i]>0 ? in[i] : 0;
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in>=0);
    return in;
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

struct PRelu : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return in>0 ? in : PRELU_FAC*in;
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    return in>0 ? 1 : PRELU_FAC;
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(VEC_WIDTH)
    for (Uint i=0;i<N; i++) out[i] = in[i]>0 ? in[i] : PRELU_FAC*in[i];
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    if(in >= 0) return in;
    else return in / PRELU_FAC;
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

struct ExpPlus : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return std::log(1+std::exp(in));
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    return 1/(1+std::exp(-in));
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    for(Uint i=0; i<N; i++) out[i] = _eval(in[i]);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    return std::log(std::exp(in)-1);
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

struct SoftPlus : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return .5*(in + std::sqrt(1+in*in));
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    return .5*(1 + in/std::sqrt(1+in*in));
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(VEC_WIDTH)
    for (Uint i=0;i<N; i++) out[i] = .5*(in[i]+std::sqrt(1+in[i]*in[i]));
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in > 0);
    return (in*in - 0.25)/in;
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

struct Exp : public Function {
  Real initFactor(const Uint inps, const Uint outs) const override {
    return std::sqrt(2./inps);
  }
  static inline nnReal _eval(const nnReal in) {
    return nnSafeExp(in);
  }
  static inline nnReal _evalDiff(const nnReal in, const nnReal d) {
    return nnSafeExp(in);
  }
  static inline void _eval(nnOpInp in, nnOpRet out, const Uint N) {
    for(Uint i=0;i<N;i++) out[i] = nnSafeExp(in[i]);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const override {
    return _eval(in, out, N);
  }
  nnReal eval(const nnReal in) const override {
    return _eval(in);
  }
  nnReal inverse(const nnReal in) const override {
    assert(in > 0);
    return std::log(in);
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override {
    return _evalDiff(in, d);
  }
};

inline Function* makeFunction(const string name, const bool bOutput=false) {
  if (bOutput || name == "Linear") return new Linear();
  else
  if (name == "Tanh")   return new Tanh();
  else
  if (name == "Sigm") return new Sigm();
  else
  if (name == "HardSign") return new HardSign();
  else
  if (name == "SoftSign") return new SoftSign();
  else
  if (name == "Relu") return new Relu();
  else
  if (name == "PRelu") return new PRelu();
  else
  if (name == "ExpPlus") return new ExpPlus();
  else
  if (name == "SoftPlus") return new SoftPlus();
  else
  if (name == "Exp") return new Exp();
  else
  die("Activation function not recognized");
  return (Function*)nullptr;
}
