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
#include <cstring>
using namespace std;

#define _allocateClean(name, size) { const int sizeSIMD=ceil(size/4.)*4.*sizeof(Real); posix_memalign((void **)& name, 32, sizeSIMD); memset(name, 0, sizeSIMD); }
#define _allocateQuick(name, size) { const int sizeSIMD=ceil(size/4.)*4.*sizeof(Real); posix_memalign((void **)& name, 32, sizeSIMD); }
#define _myfree( name ) free( name );

struct Activation //All the network signals
{
    Activation(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
    {
        //contains all inputs to each neuron (inputs to network input layer is empty)
        _allocateQuick(in_vals, nNeurons)
        //contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
        _allocateQuick(outvals, nNeurons)
        //deltas for each neuron
        _allocateQuick(errvals, nNeurons)
        //memory of LSTM
        _allocateQuick(ostates, nStates)
        //inputs to gates (cell into in_vals)
        _allocateQuick(iIGates, nStates)
        _allocateQuick(iFGates, nStates)
        _allocateQuick(iOGates, nStates)
        //output of gates and LSTM cell
        _allocateQuick(oMCell, nStates)
        _allocateQuick(oIGates, nStates)
        _allocateQuick(oFGates, nStates)
        _allocateQuick(oOGates, nStates)
        //errors of gates and LSTM cell
        _allocateQuick(eMCell, nStates)
        _allocateQuick(eIGates, nStates)
        _allocateQuick(eFGates, nStates)
        _allocateQuick(eOGates, nStates)
    }

    ~Activation()
    {
        _myfree(in_vals)
        _myfree(outvals)
        _myfree(errvals)
        _myfree(ostates)

        _myfree(iIGates)
        _myfree(iFGates)
        _myfree(iOGates)

        _myfree(oMCell)
        _myfree(oIGates)
        _myfree(oFGates)
        _myfree(oOGates)

        _myfree(eMCell)
        _myfree(eIGates)
        _myfree(eFGates)
        _myfree(eOGates)

    }

    void clearOutput()
    {
        for (int j=0; j<nNeurons; j++)
        *(outvals +j) = 0.;

        for (int j=0; j<nStates; j++)
        *(ostates +j) = 0.;
    }

    void clearErrors()
    {
        for (int j=0; j<nNeurons; j++)
            *(errvals +j) = 0.;

        for (int j=0; j<nStates; j++) {
            *(eOGates +j) = 0.;
            *(eIGates +j) = 0.;
            *(eFGates +j) = 0.;
            *(eMCell  +j) = 0.;
        }
    }

    void clearInputs()
    {
        for (int j=0; j<nNeurons; j++)
            *(in_vals +j) = 0.;

        for (int j=0; j<nStates; j++) {
            *(iIGates +j) = 0.;
            *(iFGates +j) = 0.;
            *(iOGates +j) = 0.;
        }
    }

    const int nNeurons, nStates;
    Real *in_vals, *outvals, *errvals, *ostates, *iIGates, *iFGates, *iOGates, *oMCell, *oIGates, *oFGates, *oOGates, *eMCell, *eIGates, *eFGates, *eOGates;
};

struct Grads
{
    Grads(int _nWeights, int _nBiases): nWeights(_nWeights), nBiases(_nBiases)
    {
        _allocateClean(_W, nWeights)
        _allocateClean(_B, nBiases)
    }

    ~Grads()
    {
        _myfree(_W)
        _myfree(_B)
    }

    const int nWeights, nBiases;
    Real *_W, *_B;
};

struct Mem //Memory light recipient for prediction on agents
{
    Mem(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
    {
        _allocateClean(outvals, nNeurons)
        _allocateClean(ostates, nStates)
    }

    ~Mem()
    {
        _myfree(outvals);
        _myfree(ostates);
    }
    const int nNeurons, nStates;
    Real *outvals, *ostates;
};

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
