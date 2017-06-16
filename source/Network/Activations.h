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
#define PRELU_FAC 0.1
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
	nnReal*const outvals;
	nnReal*const ostates;
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
		std::memset(outvals,0.,nNeurons*sizeof(nnReal));
		std::memset(ostates,0.,nStates*sizeof(nnReal));
		std::memset(oMCell, 0.,nStates*sizeof(nnReal));
		std::memset(oIGates,0.,nStates*sizeof(nnReal));
		std::memset(oFGates,0.,nStates*sizeof(nnReal));
		std::memset(oOGates,0.,nStates*sizeof(nnReal));
	}

	inline void clearErrors()
	{
		std::memset(errvals,0.,nNeurons*sizeof(nnReal));
		std::memset(eOGates,0.,nStates*sizeof(nnReal));
		std::memset(eIGates,0.,nStates*sizeof(nnReal));
		std::memset(eFGates,0.,nStates*sizeof(nnReal));
		std::memset(eMCell,0.,nStates*sizeof(nnReal));
	}

	inline void clearInputs()
	{
		std::memset(in_vals,0.,nNeurons*sizeof(nnReal));
		std::memset(iIGates,0.,nStates*sizeof(nnReal));
		std::memset(iFGates,0.,nStates*sizeof(nnReal));
		std::memset(iOGates,0.,nStates*sizeof(nnReal));
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
	nnReal*const in_vals;
	nnReal*const outvals;
	nnReal*const errvals;
	nnReal*const ostates;
	nnReal*const iIGates;
	nnReal*const iFGates;
	nnReal*const iOGates;
	nnReal*const oMCell;
	nnReal*const oIGates;
	nnReal*const oFGates;
	nnReal*const oOGates;
	nnReal*const eMCell;
	nnReal*const eIGates;
	nnReal*const eFGates;
	nnReal*const eOGates;
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
		std::memset(_W,0.,nWeights*sizeof(nnReal));
		std::memset(_B,0.,nBiases*sizeof(nnReal));
	}
	const Uint nWeights, nBiases;
	nnReal*const _W;
	nnReal*const _B;
};

#define ary nnReal*__restrict__ const
struct Function
{
	//weights are initialized with uniform distrib [-weightsInitFactor, weightsInitFactor]
	virtual Real weightsInitFactor(const Uint inps, const Uint outs) const = 0;
	virtual Real biasesInitFactor(const Uint outs) const
	{
		return std::numeric_limits<nnReal>::epsilon();
	}

	virtual void eval(const ary in, ary out, const Uint N) const = 0; // f(in)
	virtual nnReal eval(const nnReal in) const = 0; // f(in)
	virtual nnReal evalDiff(const nnReal in) const = 0; // f'(in)
};
//If adding a new function, edit this function readFunction at end of file

struct Linear : public Function
{
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
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
	nnReal evalDiff(const nnReal in) const override
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
			const nnReal e2x = std::exp(-2*in);
			return (1-e2x)/(1+e2x);
		} else {
			const nnReal e2x = std::exp( 2*in);
			return (e2x-1)/(1+e2x);
		}
	}
	nnReal evalDiff(const nnReal in) const override
	{
		const nnReal arg = in < 0 ? -in : in;
		const nnReal e2x = std::exp(-2.*arg);
		if (arg > EXP_CUT) return 4*e2x;
		return 4*e2x/((1+e2x)*(1+e2x));
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
		for (Uint i=0;i<N; i++) {
			const nnReal e2x = std::exp(2*in[i]);
			out[i] = (e2x-1)/(1+e2x);
		}
	}
};

struct TwoTanh : public Function
{
	Real weightsInitFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./(inps + outs));
	}
	nnReal eval(const nnReal in) const override
	{
		if(in >  EXP_CUT) return  2;
		if(in < -EXP_CUT) return -2;
		if(in>0) {
			const nnReal e2x = std::exp(-2*in);
			return 2*(1-e2x)/(1+e2x);
		} else {
			const nnReal e2x = std::exp( 2*in);
			return 2*(e2x-1)/(1+e2x);
		}
	}
	nnReal evalDiff(const nnReal in) const override
	{
		const nnReal arg = in < 0 ? -in : in;
		const nnReal e2x = arg > EXP_CUT ? std::exp(-2*EXP_CUT) : std::exp(-2.*arg);
		return 8*e2x/((1+e2x)*(1+e2x));
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
		for (Uint i=0;i<N; i++) {
			const nnReal e2x = std::exp(2*in[i]);
			out[i] = 2*(e2x-1)/(1+e2x);
		}
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
	nnReal evalDiff(const nnReal in) const override
	{
		const nnReal arg = in < 0 ? -in : in;
		const nnReal e2x = std::exp(-arg);
		if (arg > 2*EXP_CUT) return e2x;
		return e2x/((1+e2x)*(1+e2x));
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
		for (Uint i=0;i<N; i++) out[i] = 1/(1+std::exp(-in[i]));
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
	nnReal evalDiff(const nnReal in) const override
	{
		const nnReal denom = 1+std::fabs(in);
		return 1/(denom*denom);
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
		for (Uint i=0;i<N; i++) out[i] = in[i]/(1+std::fabs(in[i]));
	}
};

struct TwoSoftSign : public Function
{
	Real weightsInitFactor(const Uint inps, const Uint outs) const override
	{
		return std::sqrt(6./(inps + outs));
	}
	nnReal eval(const nnReal in) const override
	{
		return 2*in/(1+std::fabs(in));
	}
	nnReal evalDiff(const nnReal in) const override
	{
		const nnReal denom = 1+std::fabs(in);
		return 2/(denom*denom);
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
		for (Uint i=0;i<N; i++) out[i] = 2*in[i]/(1+std::fabs(in[i]));
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
	nnReal evalDiff(const nnReal in) const override
	{
		const nnReal denom = 1+std::fabs(in);
		return 0.5/(denom*denom);
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
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
	nnReal evalDiff(const nnReal in) const override
	{
		return in>0 ? 1 : 0;
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
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
	nnReal evalDiff(const nnReal in) const override
	{
		return in>0 ? 1 : PRELU_FAC;
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
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
	nnReal evalDiff(const nnReal in) const override
	{
		if(in >  2*EXP_CUT) return 1;
		if(in < -2*EXP_CUT) return std::exp(in); //neglect denom
		return 1/(1+std::exp(-in));
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
		for (Uint i=0;i<N; i++) out[i] = std::log(1+std::exp(in[i]));
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
	nnReal evalDiff(const nnReal in) const override
	{
		return .5*(1 + in/std::sqrt(1+in*in));
	}
	void eval(const ary in, ary out, const Uint N) const
	{
#pragma omp simd aligned(in,out : __vec_width__) safelen(simdWidth)
		for (Uint i=0;i<N; i++) out[i] = .5*(in[i]+std::sqrt(1+in[i]*in[i]));
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
