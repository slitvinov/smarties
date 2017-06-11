/*
 *  LSTMNet.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"
#include "../Profiler.h"
class Builder;

class Network
{
protected:
	const Uint nAgents, nThreads, nInputs, nOutputs, nLayers;
	const Uint nNeurons, nWeights, nBiases, nStates;
	const bool bDump;
public:
	const vector<Layer*> layers;
	const vector<Link*> links;
	nnReal* const weights;
	nnReal* const biases;
	nnReal* const tgt_weights;
	nnReal* const tgt_biases;
	Grads* const grad;
	const vector<Grads*> Vgrad;
	const vector<Mem*> mem;
	vector<std::mt19937>& generators;
	const vector<Uint> iOut, iInp;
	vector<Uint> dump_ID;
	const bool allocatedFrozenWeights = true;

	Uint getnWeights() const {return nWeights;}
	Uint getnBiases() const {return nBiases;}
	Uint getnOutputs() const {return nOutputs;}
	Uint getnInputs() const {return nInputs;}
	Uint getnNeurons() const {return nNeurons;}
	Uint getnStates() const {return nStates;}
	Uint getnLayers() const {return nLayers;}
	Uint getnAgents() const {return nAgents;}

	inline vector<Real> getOutputs(const Activation* const act) const
	{
		vector<Real> _output(nOutputs);
		for(Uint i=0; i<nOutputs; i++) _output[i] = *(act->outvals + iOut[i]);
		return _output;
	}
	inline vector<Real> getInputGradient(const Activation* const act) const
	{
		vector<Real> ret(nInputs);
		for(Uint j=0; j<nInputs; j++) ret[j]= act->errvals[iInp[j]];
		return ret;
	}

	Network(Builder* const B, Settings & settings) ;

	~Network()
	{
		for (auto & trash : layers) _dispose_object(trash);
		for (auto & trash : mem) _dispose_object(trash);
		for (auto & trash : Vgrad) _dispose_object(trash);
		_dispose_object( grad);
		_myfree( weights );
		_myfree( biases );
		_myfree( tgt_weights );
		_myfree( tgt_biases );
	}

	void updateFrozenWeights();
	void moveFrozenWeights(const Real alpha);
	void loadMemory(Mem * _M, Activation * _N) const;
	void clearErrors(vector<Activation*>& timeSeries) const;
	void setOutputDeltas(const vector<Real>& _errors, Activation* const _N) const;

	Activation* allocateActivation() const;
	vector<Activation*> allocateUnrolledActivations(Uint length) const;
	void deallocateUnrolledActivations(vector<Activation*>* const ret) const;
	void appendUnrolledActivations(vector<Activation*>* const ret,
			Uint length=1) const;

	void seqPredict_inputs(const vector<Real>& _input, Activation* const currActivation) const;
	void seqPredict_output(vector<Real>&_output, Activation* const currActivation) const;
	void seqPredict_execute(const vector<Activation*>& series_1, vector<Activation*>& series_2,
			const nnReal* const _weights, const nnReal* const _biases) const;
	void seqPredict_execute(const vector<Activation*>& series_1, vector<Activation*>& series_2) const
	{
		seqPredict_execute(series_1, series_2, weights, biases);
	}

	void predict(const vector<Real>& _input, vector<Real>& _output,
			vector<Activation*>& timeSeries, const Uint n_step,
			const nnReal* const _weights, const nnReal* const _biases) const;
	void predict(const vector<Real>& _input, vector<Real>& _output,
			vector<Activation*>& timeSeries, const Uint n_step) const
	{
		predict(_input, _output, timeSeries, n_step, weights, biases);
	}

	void predict(const vector<Real>& _input, vector<Real>& _output,
			Activation* const prevActivation, Activation* const currActivation,
			const nnReal* const _weights, const nnReal* const _biases) const;
	void predict(const vector<Real>& _input, vector<Real>& _output,
			Activation* const prevActivation, Activation* const currActivation) const
	{
		predict(_input, _output, prevActivation, currActivation,
				weights, biases);
	}

	void predict(const vector<Real>& _input, vector<Real>& _output,
			Activation* const net, const nnReal* const _weights,
			const nnReal* const _biases) const;
	void predict(const vector<Real>& _input, vector<Real>& _output,
			Activation* const net) const
	{
		predict(_input, _output, net, weights, biases);
	}

	void backProp(vector<Activation*>& timeSeries,
			const nnReal* const _weights, const nnReal* const biases,
			Grads* const _grads) const;
	void backProp(vector<Activation*>& timeSeries, Grads* const _grads) const
	{
		backProp(timeSeries, weights, biases, _grads);
	}

	void backProp(const vector<Real>& _errors, Activation* const net,
			const nnReal* const _weights, const nnReal* const _biases,
			Grads* const _grads) const;
	void backProp(const vector<Real>& _errors, Activation* const net,
			Grads* const _grads) const
	{
		backProp(_errors, net, weights, biases, _grads);
	}

	void checkGrads();
	inline void regularize(const Real lambda) const
	{
#pragma omp parallel for
		for (Uint j=0; j<nLayers; j++)
			layers[j]->regularize(weights, biases, lambda);
	}

	void save(vector<nnReal> & outWeights, vector<nnReal> & outBiases,
			nnReal* const _weights, nnReal* const _biases) const
	{
		for (const auto &l : layers)
			l->save(outWeights,outBiases, _weights, _biases);
	}
	void restart(vector<nnReal> & outWeights, vector<nnReal> & outBiases,
			nnReal* const _weights, nnReal* const _biases) const
	{
		for (const auto &l : layers)
			l->restart(outWeights,outBiases, _weights, _biases);
	}
	//void save(const string fname);
	void dump(const int agentID);
	//bool restart(const string fname);
};
