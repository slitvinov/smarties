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

struct Mem //Memory light recipient for prediction on agents
{
	Mem(Uint _nNeurons, Uint _nStates):
	nNeurons(_nNeurons), nStates(_nStates),
	outvals(initClean(nNeurons)), ostates(init(nStates))
	{	}

	~Mem()
	{
		_myfree(outvals);
		_myfree(ostates);
	}
	const Uint nNeurons, nStates;
	Real*const outvals;
	Real*const ostates;
};

struct Activation //All the network signals. TODO: vector of activations, one per layer, allowing classes of activations
{
	Activation(Uint _nNeurons,Uint _nStates):
	nNeurons(_nNeurons),nStates(_nStates),
	//contains all inputs to each neuron (inputs to network input layer is empty)
	in_vals(init(nNeurons)),
	//contains all neuron outputs that will be the incoming signal to linked layers (outputs of input layer is network inputs)
	outvals(init(nNeurons)),
	//deltas for each neuron
	errvals(initClean(nNeurons)),
	//memory and inputs to gates (cell into in_vals)
	ostates(init(nNeurons)), iIGates(init(nNeurons)),
	iFGates(init(nNeurons)), iOGates(init(nNeurons)),
	//output of gates and LSTM cell
	oMCell(init(nNeurons)), oIGates(init(nNeurons)),
	oFGates(init(nNeurons)), oOGates(init(nNeurons)),
	//errors of gates and LSTM cell
	eMCell(init(nNeurons)), eIGates(init(nNeurons)),
	eFGates(init(nNeurons)), eOGates(init(nNeurons))
	{ }

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

	inline void clearOutput()
	{
		std::memset(outvals,0.,nNeurons*sizeof(Real));
		std::memset(ostates,0.,nStates*sizeof(Real));
		std::memset(oMCell, 0.,nStates*sizeof(Real));
		std::memset(oIGates,0.,nStates*sizeof(Real));
		std::memset(oFGates,0.,nStates*sizeof(Real));
		std::memset(oOGates,0.,nStates*sizeof(Real));
	}

	inline void clearErrors()
	{
		std::memset(errvals,0.,nNeurons*sizeof(Real));
		std::memset(eOGates,0.,nStates*sizeof(Real));
		std::memset(eIGates,0.,nStates*sizeof(Real));
		std::memset(eFGates,0.,nStates*sizeof(Real));
		std::memset(eMCell,0.,nStates*sizeof(Real));
	}

	inline void clearInputs()
	{
		std::memset(in_vals,0.,nNeurons*sizeof(Real));
		std::memset(iIGates,0.,nStates*sizeof(Real));
		std::memset(iFGates,0.,nStates*sizeof(Real));
		std::memset(iOGates,0.,nStates*sizeof(Real));
	}

	inline void loadMemory(Mem*const _M)
	{
			assert(_M->nNeurons == nNeurons);
			assert(_M->nStates == nStates);
	    for (Uint j=0; j<nNeurons; j++) outvals[j] = _M->outvals[j];
	    for (Uint j=0; j<nStates;  j++) ostates[j] = _M->ostates[j];
	}

	inline void storeMemory(Mem*const _M)
	{
			assert(_M->nNeurons == nNeurons);
			assert(_M->nStates == nStates);
	    for (Uint j=0; j<nNeurons; j++) _M->outvals[j] = outvals[j];
	    for (Uint j=0; j<nStates;  j++) _M->ostates[j] = ostates[j];
	}

	const Uint nNeurons, nStates;
	Real*const in_vals;
	Real*const outvals;
	Real*const errvals;
	Real*const ostates;
	Real*const iIGates;
	Real*const iFGates;
	Real*const iOGates;
	Real*const oMCell;
	Real*const oIGates;
	Real*const oFGates;
	Real*const oOGates;
	Real*const eMCell;
	Real*const eIGates;
	Real*const eFGates;
	Real*const eOGates;
};

struct Grads
{
	Grads(Uint _nWeights, Uint _nBiases):
	nWeights(_nWeights), nBiases(_nBiases),
	_W(initClean(_nWeights)), _B(initClean(_nBiases))
	{ }

	~Grads()
	{
		_myfree(_W);
    _myfree(_B);
	}
	inline void clear()
	{
		std::memset(_W,0.,nWeights*sizeof(Real));
		std::memset(_B,0.,nBiases*sizeof(Real));
	}
	const Uint nWeights, nBiases;
	Real*const _W;
	Real*const _B;
};

struct Function
{
	//weights are initialized with uniform distrib [-initFactor, initFactor]
	virtual Real initFactor(const Uint inps, const Uint outs) const = 0;
	virtual Real eval(const Real in) const = 0; // f(in)
	virtual Real evalDiff(const Real in) const = 0; // f'(in)
};
//If adding a new function, edit this function readFunction at end of file

struct Linear : public Function
{
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./inps);// 2./inps;
	}
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
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./(inps + outs));
	}
	Real eval(const Real in) const override
	{
			if(in >  8) return  1;
			if(in < -8) return -1;
			if(in>0) {
				const Real e2x = std::exp(-2*in);
				return (1-e2x)/(1+e2x);
			} else {
				const Real e2x = std::exp( 2*in);
				return (e2x-1)/(1+e2x);
			}
	}

	Real evalDiff(const Real in) const override
	{
		const Real arg = in < 0 ? -in : in;
		const Real e2x = std::exp(-2.*arg);
		if (arg > 8) return 4*e2x;
		return 4*e2x/((1+e2x)*(1+e2x));
	}
};

struct TwoTanh : public Function
{
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./(inps + outs));
	}
	Real eval(const Real in) const override
	{
			if(in >  8) return  2;
			if(in < -8) return -2;
			if(in>0) {
				const Real e2x = std::exp(-2*in);
				return 2*(1-e2x)/(1+e2x);
			} else {
				const Real e2x = std::exp( 2*in);
				return 2*(e2x-1)/(1+e2x);
			}
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
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./(inps + outs));
	}
	Real eval(const Real in) const override
	{
			if(in >  16) return 1;
			if(in < -16) return 0;
			return 1/(1+std::exp(-in));
	}

	Real evalDiff(const Real in) const override
	{
		const Real arg = in < 0 ? -in : in;
		const Real e2x = std::exp(-arg);
		if (arg > 16) return e2x;
		return e2x/((1+e2x)*(1+e2x));
	}
};

struct SoftSign : public Function
{
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./(inps + outs));
	}
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
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./(inps + outs));
	}
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
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./(inps + outs));
	}
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
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return 2./inps;
	}
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
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return 2./inps;
	}
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
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return 2./inps;
	}
	Real eval(const Real in) const override
	{
			if(in >  16) return in;
			if(in < -16) return 0;
			return std::log(1+std::exp(in));
	}

	Real evalDiff(const Real in) const override
	{
		if(in >  16) return 1;
		if(in < -16) return std::exp(in); //neglect denom
		return 1/(1+std::exp(-in));
	}
};

struct SoftPlus : public Function
{
	Real initFactor(const Uint inps, const Uint outs) const override
	{
		return 2./inps;
	}
	Real eval(const Real in) const override
	{
		return .5*(in + std::sqrt(1+in*in));
	}

	Real evalDiff(const Real in) const override
	{
		return .5*(1 + in/std::sqrt(1+in*in));
	}
};

inline Function* readFunction(const string name, const bool bOutput)
{
	if (bOutput || name == "Linear") return new Linear();
	else
	if (name == "Tanh") 	return new Tanh();
	else
	if (name == "TwoTanh") return new TwoTanh();
	else
	if (name == "Sigm") return new Sigm();
	else
	if (name == "SoftSign") return new SoftSign();
	else
	if (name == "TwoSoftSign") return new TwoSoftSign();
	else
	if (name == "SoftSigm") return new SoftSigm();
	else
	if (name == "Relu") return new Relu();
	else
	if (name == "PRelu") return new PRelu();
	else
	if (name == "ExpPlus") return new ExpPlus();
	else
	if (name == "SoftPlus") return new SoftPlus();
	else
	die("Activation function not recognized\n");
	return (Function*)nullptr;
}
