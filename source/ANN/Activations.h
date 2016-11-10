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
        _allocateClean(errvals, nNeurons)
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
		_allocateClean(eMCell, nStates)
		_allocateClean(eIGates, nStates)
		_allocateClean(eFGates, nStates)
		_allocateClean(eOGates, nStates)
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
    	std::memset(outvals,0.,nNeurons);
    	std::memset(ostates,0.,nStates);
    }

    void clearErrors()
    {
    	std::memset(errvals,0.,nNeurons);
    	std::memset(eOGates,0.,nStates);
    	std::memset(eIGates,0.,nStates);
    	std::memset(eFGates,0.,nStates);
    	std::memset(eMCell,0.,nStates);
    }

    void clearInputs()
    {
    	std::memset(in_vals,0.,nNeurons);
    	std::memset(iIGates,0.,nStates);
    	std::memset(iFGates,0.,nStates);
    	std::memset(iOGates,0.,nStates);
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

struct Linear
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = in[o];
    }
    
	inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = 1.;
    }
	
	inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		return;
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = err[o];
    }
};
struct Tanh
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real e2x = std::exp(2.*in[o]);
			out[o] = (e2x - 1.) / (e2x + 1.);
		}
    }
    
	inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real e2x = std::exp(2.*in[o]);
			const Real t = (e2x + 1.);
			out[o]  = 4*e2x/(t*t);
		}
    }
    
	inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real e2x = std::exp(2.*in[o]);
			const Real t = (e2x + 1.);
			out[o] *= 4*e2x/(t*t);
		}
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real e2x = std::exp(2.*in[o]);
			const Real t = (e2x + 1.);
			out[o] = 4*err[o]*e2x/(t*t);
		}
    }
};

struct TwoTanh
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real e2x = std::exp(2.*in[o]);
			out[o] = 2.*(e2x - 1.) / (e2x + 1.);
		}
    }
    
	inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real e2x = std::exp(2.*in[o]);
			const Real t = (e2x + 1.);
			out[o]  = 8.*e2x/(t*t);
		}
    }
    
	inline static void mullDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real e2x = std::exp(2.*in[o]);
			const Real t = (e2x + 1.);
			out[o] *= 8.*e2x/(t*t);
		}
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real e2x = std::exp(2.*in[o]);
			const Real t = (e2x + 1.);
			out[o] = 8*err[o]*e2x/(t*t);
		}
    }
};
struct Sigm
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = 1. / (1. + std::exp(-in[o]));
    }
    
	inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real ex = std::exp(in[o]);
			const Real e2x = (1. + ex)*(1. + ex);
			out[o]  = ex/e2x;
		}
    }
    
	inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real ex = std::exp(in[o]);
			const Real e2x = (1. + ex)*(1. + ex);
			out[o] *= ex/e2x;
		}
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real ex = std::exp(in[o]);
			const Real e2x = (1. + ex)*(1. + ex);
			out[o] = err[o]*ex/e2x;
		}
    }
};
struct SoftSign
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = in[o]/(1. + std::fabs(in[o]));
    }
    inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1. + std::fabs(in[o]);
			out[o]  = 1./(denom*denom);
		}
    }
    inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1. + std::fabs(in[o]);
			out[o] *= 1./(denom*denom);
		}
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1. + std::fabs(in[o]);
			out[o] = err[o]/(denom*denom);
		}
    }
};
struct TwoSoftSign
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = 2*in[o]/(1. + std::fabs(in[o]));
    }
	inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1. + std::fabs(in[o]);
			out[o]  = 2./(denom*denom);
		}
    }
	inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1. + std::fabs(in[o]);
			out[o] *= 2./(denom*denom);
		}
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1. + std::fabs(in[o]);
			out[o] = 2*err[o]/(denom*denom);
		}
    }
};
struct SoftSigm
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = 0.5*(1. + in[o]/(1.+std::fabs(in[o])));
    }
    inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1.+std::fabs(in[o]);
			out[o]  = 0.5/(denom*denom);
		}
    }
    inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1.+std::fabs(in[o]);
			out[o] *= 0.5/(denom*denom);
		}
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1.+std::fabs(in[o]);
			out[o] = 0.5*err[o]/(denom*denom);
		}
    }
};
struct HardSign
{
    inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = in[o]/std::sqrt(1. + in[o]*in[o]);
    }
    inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1./std::sqrt(1. + in[o]*in[o]);
			out[o]  = denom*denom*denom;
		}
    }
    inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1./std::sqrt(1. + in[o]*in[o]);
			out[o] *= denom*denom*denom;
		}
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1./std::sqrt(1. + in[o]*in[o]);
			out[o] = err[o]*denom*denom*denom;
		}
    }
};
struct TwoHardSign
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
	{
		for (int o=0; o<N; o++) out[o] = 2*in[o]/std::sqrt(1. + in[o]*in[o]);
	}
	inline static void evalDiff(const Real* const in, Real* const out, const int& N)
	{
		for (int o=0; o<N; o++) {
			const Real denom = 1./std::sqrt(1. + in[o]*in[o]);
			out[o]  = 2*denom*denom*denom;
		}
	}
	inline static void mulDiff(const Real* const in, Real* const out, const int& N)
	{
		for (int o=0; o<N; o++) {
			const Real denom = 1./std::sqrt(1. + in[o]*in[o]);
			out[o] *= 2*denom*denom*denom;
		}
	}
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1./std::sqrt(1. + in[o]*in[o]);
			out[o] = 2*err[o]*denom*denom*denom;
		}
    }
};
struct HardSigm
{
    inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = 0.5*(1. + in[o]/std::sqrt(1. + in[o]*in[o]));
    }
    inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1/std::sqrt(1. + in[o]*in[o]);
			out[o]  = 0.5*denom*denom*denom;
		}
    }
    inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1/std::sqrt(1. + in[o]*in[o]);
			out[o] *= 0.5*denom*denom*denom;
		}
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) {
			const Real denom = 1/std::sqrt(1. + in[o]*in[o]);
			out[o] = 0.5*err[o]*denom*denom*denom;
		}
    }
};
struct Relu
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = in[o]>0 ? in[o] : 0;
    }
    inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = in[o]>0 ? 1.0 : 0;
    }
    inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = in[o]>0 ? out[o] : 0;
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = in[o]>0 ? err[o] : 0;
    }
};
struct ExpPlus
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = log(1.+std::exp(in[o]));
    }
    inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o]  = 1./(1. + std::exp(-in[o]));
    }
    inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] *= 1./(1. + std::exp(-in[o]));
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = err[o]/(1. + std::exp(-in[o]));
    }
};

struct SoftPlus
{
	inline static void eval(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = .5*(in[o]+std::sqrt(1+in[o]*in[o]));
    }
    inline static void evalDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o]  = .5*(1.+in[o]/std::sqrt(1+in[o]*in[o]));
    }
    inline static void mulDiff(const Real* const in, Real* const out, const int& N)
    {
		for (int o=0; o<N; o++) out[o] *= .5*(1.+in[o]/std::sqrt(1+in[o]*in[o]));
    }
	
	inline static void evalDiff(const Real* const in, Real* const out, const Real* const err, const int& N)
    {
		for (int o=0; o<N; o++) out[o] = .5*err[o]*(1.+in[o]/std::sqrt(1+in[o]*in[o]));
    }
};
