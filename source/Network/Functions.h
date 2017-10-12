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

//List of non-linearities for neural networks
//- eval return f(in), also present as array in / array out
//- evalDiff returns f'(x)
//- weightsInitFactor: some prefer fan in fan out, some only fan-in dependency
//If adding a new function, edit function readFunction at end of file

struct Function
{
  //weights are initialized with uniform distrib [-weightsInitFactor, weightsInitFactor]
  virtual Real weightsInitFactor(const Uint inps, const Uint outs) const = 0;
  virtual Real biasesInitFactor(const Uint outs) const
  {
    return std::numeric_limits<nnReal>::epsilon();
  }

  virtual void eval(nnOpInp in, nnOpRet out, const Uint N) const = 0; // f(in)
  virtual nnReal eval(const nnReal in) const = 0; // f(in)
  virtual nnReal evalDiff(const nnReal in, const nnReal d) const = 0; // f'(in)
  virtual ~Function() {}
};

struct Linear : public Function
{
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(simdWidth)
    for (Uint i=0;i<N; i++) out[i] = in[i];
  }
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);// 2./inps;
  }
  nnReal eval(const nnReal in) const override
  {
    return in;
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    return 1;
  }
};

struct Tanh : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }
  nnReal eval(const nnReal in) const override
  {
    if(in >   EXP_CUT) return  1;
    if(in < - EXP_CUT) return -1;
    if(in>0) {
      const nnReal e2x = nnSafeExp(-2*in);
      return (1-e2x)/(1+e2x);
    } else {
      const nnReal e2x = nnSafeExp( 2*in);
      return (e2x-1)/(1+e2x);
    }
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    const nnReal arg = in < 0 ? -in : in; //symmetric
    const nnReal e2x = nnSafeExp(-2.*arg);
    if (arg > EXP_CUT && d*in > 0) return 0;
    if (arg > EXP_CUT) return 4*e2x; //in*d > 0 ? 0 :
    return 4*e2x/((1+e2x)*(1+e2x));
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    for (Uint i=0;i<N; i++)  out[i] = eval(in[i]);
  }
};

struct Sigm : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }
  nnReal eval(const nnReal in) const override
  {
    if(in >  2*EXP_CUT) return 1;
    if(in < -2*EXP_CUT) return 0;
    return 1/(1+std::exp(-in));
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    const nnReal arg = in < 0 ? -in : in;
    const nnReal e2x = std::exp(-arg);
    if (arg > 2*EXP_CUT) return e2x;
    return e2x/((1+e2x)*(1+e2x));
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(simdWidth)
    for (Uint i=0;i<N; i++) out[i] = 1/(1+std::exp(-in[i]));
  }
};

struct HardSign : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }
  nnReal eval(const nnReal in) const override
  {
    return in/std::sqrt(2+in*in);
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    const nnReal denom = std::sqrt(2+in*in);
    return 2/(denom*denom*denom);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(simdWidth)
    for (Uint i=0;i<N; i++) out[i] = in[i]/std::sqrt(2+in[i]*in[i]);
  }
};

struct SoftSign : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }
  nnReal eval(const nnReal in) const override
  {
    return in/(1+std::fabs(in));
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    const nnReal denom = 1+std::fabs(in);
    return 1/(denom*denom);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(simdWidth)
    for (Uint i=0;i<N; i++) out[i] = in[i]/(1+std::fabs(in[i]));
  }
};

struct SoftSigm : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(6./(inps + outs));
  }
  nnReal eval(const nnReal in) const override
  {
    const nnReal sign = in/(1+std::fabs(in));
    return 0.5*(1+sign);
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    const nnReal denom = 1+std::fabs(in);
    return 0.5/(denom*denom);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(simdWidth)
    for (Uint i=0;i<N; i++) out[i] = 0.5*(1+in[i]/(1+std::fabs(in[i])));
  }
};

struct Relu : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);
  }
  nnReal eval(const nnReal in) const override
  {
    return in>0 ? in : 0;
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    return in>0 ? 1 : 0;
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(simdWidth)
    for (Uint i=0;i<N; i++) out[i] = in[i]>0 ? in[i] : 0;
  }
};

struct PRelu : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);
  }
  nnReal eval(const nnReal in) const override
  {
    return in>0 ? in : PRELU_FAC*in;
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    return in>0 ? 1 : PRELU_FAC;
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(simdWidth)
    for (Uint i=0;i<N; i++) out[i] = in[i]>0 ? in[i] : PRELU_FAC*in[i];
  }
};

struct ExpPlus : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);
  }
  nnReal eval(const nnReal in) const override
  {
    if(in >  2*EXP_CUT) return in;
    if(in < -2*EXP_CUT) return 0;
    return std::log(1+std::exp(in));
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    if(in >  2*EXP_CUT) return 1;
    if(in < -2*EXP_CUT) return nnSafeExp(in); //neglect denom
    return 1/(1+std::exp(-in));
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    for (Uint i=0;i<N; i++) out[i] = std::log(1+nnSafeExp(in[i]));
  }
};

struct SoftPlus : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);
  }
  nnReal eval(const nnReal in) const override
  {
    return .5*(in + std::sqrt(1+in*in));
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    return .5*(1 + in/std::sqrt(1+in*in));
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    #pragma omp simd aligned(in,out : VEC_WIDTH) safelen(simdWidth)
    for (Uint i=0;i<N; i++) out[i] = .5*(in[i]+std::sqrt(1+in[i]*in[i]));
  }
};

struct Exp : public Function
{
  Real weightsInitFactor(const Uint inps, const Uint outs) const override
  {
    return std::sqrt(2./inps);
  }
  nnReal eval(const nnReal in) const override
  {
    return nnSafeExp(in);
  }
  nnReal evalDiff(const nnReal in, const nnReal d) const override
  {
    return nnSafeExp(in);
  }
  void eval(nnOpInp in, nnOpRet out, const Uint N) const
  {
    for(Uint i=0;i<N;i++) out[i] = nnSafeExp(in[i]);
  }
};

inline Function* readFunction(const string name, const bool bOutput=false)
{
  if (bOutput || name == "Linear") return new Linear();
  else
  if (name == "Tanh")   return new Tanh();
  else
  if (name == "Sigm") return new Sigm();
  else
  if (name == "SoftSign") return new SoftSign();
  else
  if (name == "SoftSigm") return new SoftSigm();
  else
  if (name == "Relu") return new Relu();
  else
  if (name == "PRelu") return new PRelu();
  else
  if (name == "ExpPlus") return new ExpPlus();
  else
  if (name == "HardSign") return new HardSign();
  else
  if (name == "SoftPlus") return new SoftPlus();
  else
  if (name == "Exp") return new Exp();
  else
  die("Activation function not recognized\n");
  return (Function*)nullptr;
}
