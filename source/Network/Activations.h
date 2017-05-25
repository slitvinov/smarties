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
using namespace std;
#ifndef PRELU_FAC
#define PRELU_FAC 0.001
#endif
//#define _allocateClean(name, size) { const int nsimd = __vec_width__/sizeof(Real); const int sizeSIMD=std::ceil(size/(Real)nsimd)*nsimd*sizeof(Real); posix_memalign((void **)& name, __vec_width__, sizeSIMD); memset(name, 0, sizeSIMD); }
//#define _allocateQuick(name, size) { const int nsimd = __vec_width__/sizeof(Real); const int sizeSIMD=std::ceil(size/(Real)nsimd)*nsimd*sizeof(Real); posix_memalign((void **)& name, __vec_width__, sizeSIMD); }
//#define _myfree( name ) free( name );

struct Activation //All the network signals
{
	Activation(int _nNeurons,int _nStates):nNeurons(_nNeurons),nStates(_nStates)
	{
		//contains all inputs to each neuron (inputs to network input layer is empty)
		_allocateQuick(in_vals, nNeurons);
		//contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
		_allocateQuick(outvals, nNeurons);
		//deltas for each neuron
		_allocateClean(errvals, nNeurons);
		//memory of LSTM
		_allocateQuick(ostates, nStates);
		//inputs to gates (cell into in_vals)
		_allocateQuick(iIGates, nStates);
		_allocateQuick(iFGates, nStates);
		_allocateQuick(iOGates, nStates);
		//output of gates and LSTM cell
		_allocateQuick(oMCell, nStates);
		_allocateQuick(oIGates, nStates);
		_allocateQuick(oFGates, nStates);
		_allocateQuick(oOGates, nStates);
		//errors of gates and LSTM cell
		_allocateClean(eMCell, nStates);
		_allocateClean(eIGates, nStates);
		_allocateClean(eFGates, nStates);
		_allocateClean(eOGates, nStates);
	}

	~Activation()
	{
		_myfree(in_vals);
		_myfree(outvals);
		_myfree(errvals);
		_myfree(ostates);

		_myfree(iIGates);
		_myfree(iFGates);
		_myfree(iOGates);

		_myfree(oMCell);
		_myfree(oIGates);
		_myfree(oFGates);
		_myfree(oOGates);

		_myfree(eMCell);
		_myfree(eIGates);
		_myfree(eFGates);
		_myfree(eOGates);
	}

	void clearOutput()
	{
		std::memset(outvals,0.,nNeurons);
		std::memset(ostates,0.,nStates);
		std::memset(oMCell, 0.,nStates);
		std::memset(oIGates,0.,nStates);
		std::memset(oFGates,0.,nStates);
		std::memset(oOGates,0.,nStates);
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
	Real *in_vals, *outvals, *errvals, *ostates;
	Real *iIGates, *iFGates, *iOGates;
	Real *oMCell, *oIGates, *oFGates, *oOGates;
	Real *eMCell, *eIGates, *eFGates, *eOGates;
};

struct Grads
{
	Grads(int _nWeights, int _nBiases): nWeights(_nWeights), nBiases(_nBiases)
	{
		_allocateClean(_W, nWeights);
        		_allocateClean(_B, nBiases);
	}

	~Grads()
	{
		_myfree(_W);
        		_myfree(_B);
	}
	void clear()
	{
		std::memset(_W,0.,nWeights);
		std::memset(_B,0.,nBiases);
	}
	const int nWeights, nBiases;
	Real *_W, *_B;
};

struct Mem //Memory light recipient for prediction on agents
{
	Mem(int _nNeurons, int _nStates): nNeurons(_nNeurons), nStates(_nStates)
	{
		_allocateClean(outvals, nNeurons);
        		_allocateClean(ostates, nStates);
	}

	~Mem()
	{
		_myfree(outvals);
		_myfree(ostates);
	}
	const int nNeurons, nStates;
	Real *outvals, *ostates;
};

struct Function
{
	virtual Real eval(const Real in) const = 0;
	virtual Real evalDiff(const Real in) const = 0;
};

struct Linear : public Function
{
	Real eval(const Real in) const override
	{
		return in;
	}

	Real evalDiff(const Real in) const override
	{
		return 1;
	}
};

struct Tanh : public Function
{
	Real eval(const Real in) const override
	{
			if(in >  8) return  1;
			if(in < -8) return -1;
			const Real e2x = std::exp(-2*in);
			return (1-e2x)/(1+e2x);
	}

	Real evalDiff(const Real in) const override
	{
		const Real arg = in < 0 ? -in : in;
		const Real e2x = arg > 8 ? std::exp(-16) : std::exp(-2.*arg);
		return 4*e2x/((1+e2x)*(1+e2x));
	}
};

struct TwoTanh : public Function
{
	Real eval(const Real in) const override
	{
			if(in >  8) return  2;
			if(in < -8) return -2;
			const Real e2x = std::exp(-2*in);
			return 2*(1-e2x)/(1+e2x);
	}

	Real evalDiff(const Real in) const override
	{
		const Real arg = in < 0 ? -in : in;
		const Real e2x = arg > 8 ? std::exp(-16) : std::exp(-2.*arg);
		return 8*e2x/((1+e2x)*(1+e2x));
	}
};

struct Sigm : public Function
{
	Real eval(const Real in) const override
	{
			if(in >  8) return 1;
			if(in < -8) return 0;
			return 1/(1+std::exp(-in));
	}

	Real evalDiff(const Real in) const override
	{
		const Real arg = in < 0 ? -in : in;
		const Real e2x = arg > 8 ? std::exp(-8) : std::exp(-arg);
		return e2x/((1+e2x)*(1+e2x));
	}
};

struct SoftSign : public Function
{
	Real eval(const Real in) const override
	{
			return in/(1+std::fabs(in));
	}

	Real evalDiff(const Real in) const override
	{
		const Real denom = 1+std::fabs(in);
		return 1/(denom*denom);
	}
};

struct TwoSoftSign : public Function
{
	Real eval(const Real in) const override
	{
			return 2*in/(1+std::fabs(in));
	}

	Real evalDiff(const Real in) const override
	{
		const Real denom = 1+std::fabs(in);
		return 2/(denom*denom);
	}
};

struct SoftSigm : public Function
{
	Real eval(const Real in) const override
	{
			const Real sign = in/(1+std::fabs(in));
			return 0.5*(1+sign);
	}

	Real evalDiff(const Real in) const override
	{
		const Real denom = 1+std::fabs(in);
		return 0.5/(denom*denom);
	}
};

struct Relu : public Function
{
	Real eval(const Real in) const override
	{
			return in>0 ? in : 0;
	}

	Real evalDiff(const Real in) const override
	{
			return in>0 ? 1 : 0;
	}
};

struct PRelu : public Function
{
	Real eval(const Real in) const override
	{
			return in>0 ? in : PRELU_FAC*in;
	}

	Real evalDiff(const Real in) const override
	{
			return in>0 ? 1 : PRELU_FAC;
	}
};

struct ExpPlus : public Function
{
	Real eval(const Real in) const override
	{
			if(in >  8) return in;
			if(in < -8) return 0;
			return std::log(1+std::exp(in));
	}

	Real evalDiff(const Real in) const override
	{
		if(in >  8) return 1;
		if(in < -8) return std::exp(-in); //neglect denom
		return 1/(1+std::exp(-in));
	}
};

struct SoftPlus : public Function
{
	Real eval(const Real in) const override
	{
		return .5*(in + std::sqrt(1+in*in));
	}

	Real evalDiff(const Real in) const override
	{
		return .5*(1 + in/std::sqrt(1+in*in));
	}
};
