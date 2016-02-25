/*
 *  Activations.h
 *  rl
 *
 *  Created by Guido Novati on 04.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <cmath>

#include "../Settings.h"
using namespace std;

class ActivationFunction
{
public:
    virtual inline Real eval(const Real& arg) { return 0.;};
    virtual inline Real evalDiff(const Real& arg) { return 0.;};
    virtual inline Real eval(const Real * arg) { return eval(*(arg));};
    virtual inline Real evalDiff(const Real * arg) { return evalDiff(*(arg));};
#if SIMD != 1
    virtual inline vec eval(const vec & arg) { die("Wrong activation function.\n"); return arg;};
    virtual inline vec evalDiff(const vec & arg) { die("Wrong activation function.\n"); return arg;};
#endif
};
class Tanh : public ActivationFunction
{
public:
    inline Real eval(const Real& arg)
    {
        if (arg > 20)  return 1;
        if (arg < -20) return -1;
        Real e2x = exp(2.*arg);
        return (e2x - 1.) / (e2x + 1.);
    }
    
    inline Real evalDiff(const Real& arg)
    {
        //if (arg > 20 || arg < -20) return 0;
        
        Real e2x = exp(2.*arg);
        Real t = (e2x + 1.);
        return 4*e2x/(t*t);
    }
};
class Tanh2 : public ActivationFunction
{
public:
    inline Real eval(const Real& arg)
    {
        if (arg > 20)  return 2;
        if (arg < -20) return -2;
        Real e2x = exp(2.*arg);
        return 2.*(e2x - 1.) / (e2x + 1.);
    }
    
    inline Real evalDiff(const Real& arg)
    {
        //if (arg > 20 || arg < -20) return 0;
        
        Real e2x = exp(2.*arg);
        Real t = (e2x + 1.);
        return 8.*e2x/(t*t);
    }
};
class Sigm : public ActivationFunction
{
public:
    inline Real eval(const Real& arg)
    {
        //if (arg > 20)  return 1;
        //if (arg < -20) return 0;
        
        return 1. / (1. + exp(-arg));
    }
    
    inline Real evalDiff(const Real& arg)
    {
        //if (arg > 20 || arg < -20) return 0;
        
        Real ex = exp(arg);
        Real e2x = (1. + ex)*(1. + ex);
        
        return ex/e2x;
    }
};
class Linear : public ActivationFunction
{
public:
    inline Real eval(const Real& arg)
    {
        return arg;
    }
    
    inline Real evalDiff(const Real& arg)
    {
        return 1;
    }
#if SIMD != 1
    inline vec eval(const vec& arg)
    {
        return arg;
    }
    inline vec evalDiff(const vec& arg)
    {
        return SET1(1.);
    }
#endif
};
class Gaussian : public ActivationFunction
{
public:
    inline Real eval(const Real& x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        if (x > 5 || x < -5) return 0;
        return exp(-10.*x*x);
    }
    inline Real evalDiff(const Real& x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        if (x > 5 || x < -5) return 0;
        return -20. * x * exp(-10.*x*x);
    }
};
class SoftSign : public ActivationFunction
{
public:
    inline Real eval(const Real& x)
    {
        return x/(1. + fabs(x));
    }
    inline Real evalDiff(const Real& x)
    {
        Real denom = 1. + fabs(x);
        return 1./(denom*denom);
    }
};
class SoftSign2 : public ActivationFunction
{
public:
    inline Real eval(const Real& x)
    {
        return 2*x/(1. + fabs(x));
    }
    inline Real evalDiff(const Real& x)
    {
        Real denom = 1. + fabs(x);
        return 2./(denom*denom);
    }
};
class SoftSigm : public ActivationFunction
{
public:
    inline Real eval(const Real& x)
    {
        Real _x = 2*x;
        return 0.5*(1. + _x/(1. + fabs(_x)));
    }
    inline Real evalDiff(const Real& x)
    {
        Real denom = 1. + 2*fabs(x);
        return 1./(denom*denom);
    }
};
class HardSign : public ActivationFunction
{
    Real a;
public:
    HardSign(Real a = 1) : a(a) {}
    inline Real eval(const Real& x)
    {
        return a*x/sqrt(1. + a*a*x*x);
    }
    inline Real evalDiff(const Real& x)
    {
        Real denom = 1./sqrt(1. + a*a*x*x);
        return a*(denom*denom*denom);
    }
#if SIMD != 1 //TODO a
    inline vec eval(const vec& x)
    {
        return MUL(x, RSQRT( MUL(x,x)));
    }
    inline vec evalDiff(const vec& x)
    {
        const vec tmp = RSQRT( MUL (x,x));
        return MUL(tmp, MUL(tmp,tmp));
    }
#endif
};
class HardSigm : public ActivationFunction
{
    Real a;
public:
    HardSigm(Real a = 1) : a(a) {}
    inline Real eval(const Real& x)
    {
        return 0.5*(1. + a*x/sqrt(1. + a*a*x*x) );
    }
    inline Real evalDiff(const Real& x)
    {
        Real denom = 1/sqrt(1. + a*a*x*x);
        return a*0.5*(denom*denom*denom);
    }
#if SIMD != 1
    inline vec eval(const vec& x)
    {
        return MUL( SET1(0.5), ADD(SET1(1.), MUL(x, RSQRT( MUL(x,x))) ));
    }
    inline vec evalDiff(const vec& x)
    {
        const vec tmp = RSQRT( MUL (x,x));
        const vec tmp2 = MUL (SET1(0.5), tmp);
        return MUL(tmp2, MUL(tmp,tmp) );
    }
#endif
};

