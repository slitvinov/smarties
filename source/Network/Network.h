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
#include <vector>
#include <functional>
#include "../Profiler.h"

#include <fstream>
class Builder;

class Network
{
protected:
		const int nAgents, nThreads, nInputs, nOutputs, nLayers;
		const int nNeurons, nWeights, nBiases, nStates;
		const bool bDump;
    const vector<int> iOut;
    const vector<int> iInp;
    const vector<Layer*> layers;
		const vector<Link*> links;
public:
		Real* const weights;
		Real* const biases;
		Real* const tgt_weights;
		Real* const tgt_biases;
		Grads* const grad;
		const vector<Grads*> Vgrad;
		vector<std::mt19937>& generators;
		vector<int> dump_ID;
    vector<Mem*> mem;
		const bool allocatedFrozenWeights = true;

    int getnWeights() const {return nWeights;}
    int getnBiases() const {return nBiases;}
    int getnOutputs() const {return nOutputs;}
    int getnInputs() const {return nInputs;}
    int getnNeurons() const {return nNeurons;}
    int getnStates() const {return nStates;}
    int getnLayers() const {return nLayers;}
    int getnAgents() const {return nAgents;}

    vector<Real> getOutputs(const Activation* const act)
		{
		  vector<Real> _output(nOutputs);
		  for (int i=0; i<nOutputs; i++)
		      _output[i] = *(act->outvals + iOut[i]);
		  return _output;
		}

    Network(Builder* const B, Settings & settings) ;

    ~Network()
    {
        for (auto & trash : layers) _dispose_object( trash);
        for (auto & trash : mem) _dispose_object( trash);
				for (auto & trash : Vgrad) _dispose_object( trash);
        _dispose_object( grad);
        _myfree( weights )
        _myfree( biases )
        _myfree( tgt_weights )
        _myfree( tgt_biases )
    }

    void updateFrozenWeights();
    void moveFrozenWeights(const Real alpha);
    void loadMemory(Mem * _M, Activation * _N) const;
    void clearErrors(vector<Activation*>& timeSeries) const;
    void setOutputDeltas(const vector<Real>& _errors, Activation* const _N) const;

    Activation* allocateActivation() const;
    vector<Activation*> allocateUnrolledActivations(int length) const;
    void deallocateUnrolledActivations(vector<Activation*>* const ret) const;
    void appendUnrolledActivations(vector<Activation*>* const ret,
																	 int length=1) const;

    void seqPredict_inputs(const vector<Real>& _input, Activation* const currActivation) const;
    void seqPredict_output(vector<Real>&_output, Activation* const currActivation) const;
    void seqPredict_execute(const vector<Activation*>& series_1, vector<Activation*>& series_2,
    		const Real* const _weights, const Real* const _biases) const;
    void seqPredict_execute(const vector<Activation*>& series_1, vector<Activation*>& series_2) const
    {
    	seqPredict_execute(series_1, series_2, weights, biases);
    }

    void predict(const vector<Real>& _input, vector<Real>& _output,
  							 vector<Activation*>& timeSeries, const int n_step,
							 	 const Real* const _weights, const Real* const _biases) const;
    void predict(const vector<Real>& _input, vector<Real>& _output,
    						 vector<Activation*>& timeSeries, const int n_step) const
    {
        predict(_input, _output, timeSeries, n_step, weights, biases);
    }

    void predict(const vector<Real>& _input, vector<Real>& _output,
						Activation* const prevActivation, Activation* const currActivation,
						const Real* const _weights, const Real* const _biases) const;
    void predict(const vector<Real>& _input, vector<Real>& _output,
						Activation* const prevActivation, Activation* const currActivation) const
    {
        predict(_input, _output, prevActivation, currActivation,
								weights, biases);
    }

    void predict(const vector<Real>& _input, vector<Real>& _output,
								 Activation* const net, const Real* const _weights,
						  	 const Real* const _biases) const;
    void predict(const vector<Real>& _input, vector<Real>& _output,
									Activation* const net) const
    {
        predict(_input, _output, net, weights, biases);
    }

    void backProp(vector<Activation*>& timeSeries,
    							const Real* const _weights, const Real* const biases,
									Grads* const _grads) const;
    void backProp(vector<Activation*>& timeSeries, Grads* const _grads) const
    {
    	backProp(timeSeries, weights, biases, _grads);
    }

    void backProp(const vector<Real>& _errors, Activation* const net,
    							const Real* const _weights, const Real* const biases,
									Grads* const _grads) const;
    void backProp(const vector<Real>& _errors, Activation* const net,
									Grads* const _grads) const
    {
    	backProp(_errors, net, weights, biases, _grads);
    }

    void checkGrads(const vector<vector<Real>>& inputs, int lastn=-1);
		void regularize(const Real lambda)
		{
			#pragma omp parallel for
			for (int j=0; j<nLayers; j++)
	        layers[j]->regularize(weights, biases, lambda);
		}

    void save(const string fname);
    void dump(const int agentID);
    bool restart(const string fname);
};
