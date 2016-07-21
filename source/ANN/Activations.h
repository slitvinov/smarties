/*
 *  Activations.h
 *  rl
 *
 *  Created by Dmitry Alexeev and extended by Guido Novati on 04.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <cmath>

#include "../Settings.h"
using namespace std;

class Response
{
public:
    virtual inline Real eval(const Real& arg) const
    {
        return arg;
    }
    
    virtual inline Real evalDiff(const Real& arg) const
    {
        return 1;
    }
};
class Tanh : public Response
{
public:
    inline Real eval(const Real& arg) const override
    {
        if (arg >  20.) return  1.;
        if (arg < -20.) return -1.;
        const Real e2x = exp(2.*arg);
        return (e2x - 1.) / (e2x + 1.);
    }
    
    inline Real evalDiff(const Real& arg) const override
    {
        const Real e2x = exp(2.*arg);
        const Real t = (e2x + 1.);
        return 4*e2x/(t*t);
    }
};
class Tanh2 : public Response
{
public:
    inline Real eval(const Real& arg) const override
    {
        if (arg >  20.) return  2.;
        if (arg < -20.) return -2.;
        const Real e2x = exp(2.*arg);
        return 2.*(e2x - 1.) / (e2x + 1.);
    }
    
    inline Real evalDiff(const Real& arg) const override
    {
        const Real e2x = exp(2.*arg);
        const Real t = (e2x + 1.);
        return 8.*e2x/(t*t);
    }
};
class Sigm : public Response
{
public:
    inline Real eval(const Real& arg) const override
    {
        if (arg >  10.) return 1.;
        if (arg < -10.) return 0.;
        return 1. / (1. + exp(-arg));
    }
    
    inline Real evalDiff(const Real& arg) const override
    {
        const Real ex = exp(arg);
        const Real e2x = (1. + ex)*(1. + ex);
        return ex/e2x;
    }
};
class Gaussian : public Response
{
public:
    inline Real eval(const Real& x) const override
    {
        if (x > 3 || x < -3) return 0;
        return exp(-x*x);
    }
    inline Real evalDiff(const Real& x) const override
    {
        return -2. * x * exp(-x*x);
    }
};
class SoftSign : public Response
{
public:
    inline Real eval(const Real& x) const override
    {
        return x/(1. + fabs(x));
    }
    inline Real evalDiff(const Real& x) const override
    {
        const Real denom = 1. + fabs(x);
        return 1./(denom*denom);
    }
};
class SoftSign2 : public Response
{
public:
    inline Real eval(const Real& x) const override
    {
        return 2*x/(1. + fabs(x));
    }
    inline Real evalDiff(const Real& x) const override
    {
        const Real denom = 1. + fabs(x);
        return 2./(denom*denom);
    }
};
class SoftSigm : public Response
{
public:
    inline Real eval(const Real& x) const override
    {
        const Real _x = 2*x;
        return 0.5*(1. + _x/(1. + fabs(_x)));
    }
    inline Real evalDiff(const Real& x) const override
    {
        const Real denom = 1. + 2*fabs(x);
        return 1./(denom*denom);
    }
};
class HardSign : public Response
{
    const Real a;
public:
    HardSign(Real a = 1) : a(a) {}
    inline Real eval(const Real& x) const override
    {
        return a*x/sqrt(1. + a*a*x*x);
    }
    inline Real evalDiff(const Real& x) const override
    {
        const Real denom = 1./sqrt(1. + a*a*x*x);
        return a*(denom*denom*denom);
    }
};
class HardSigm : public Response
{
    const Real a;
public:
    HardSigm(Real a = 1) : a(a) {}
    inline Real eval(const Real& x) const override
    {
        return 0.5*(1. + a*x/sqrt(1. + a*a*x*x) );
    }
    inline Real evalDiff(const Real& x) const override
    {
        const Real denom = 1/sqrt(1. + a*a*x*x);
        return a*0.5*(denom*denom*denom);
    }
};