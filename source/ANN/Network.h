/*
 *  LSTMNet.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#define __WHITEN_DEFAULT false
//#define __WHITEN_DEFAULT true
#include <vector>
#include <functional>
#include "Layers.h"
#include "../Profiler.h"

#include <fstream>
using namespace std;

class Network
{
protected:
	bool bBuilt, bAddedInput;
    const Real Pdrop; //dropout
    vector<int> iOut, dump_ID;
    vector<Graph*> G;
    vector<Layer*> layers;
    void build_LSTM_layer(Graph* const graph);
    void build_normal_layer(Graph* const graph);
    void build_whitening_layer(Graph* const graph);
    void build_conv2d_layer(Graph* const graph);

public:
    int nAgents, nThreads, nInputs, nOutputs, nLayers, nNeurons, nWeights, nBiases, nStates;
    bool allocatedFrozenWeights, allocatedDroputWeights, backedUp, bDump;
    mt19937 * gen;
    vector<Mem*> mem;
    Grads * grad;//, * _grad;
    Real *weights, *biases, *tgt_weights, *tgt_biases, *weights_DropoutBackup, *running_std, *running_avg;
    vector<Grads*> Vgrad;

    int counter, batch_counter;

    void build();
    
    int getnWeights() const {assert(bBuilt); return nWeights;}
    int getnBiases() const {assert(bBuilt); return nBiases;}
    int getnOutputs() const {assert(bBuilt); return nOutputs;}
    int getnInputs() const {assert(bBuilt); return nInputs;}
    int getLastLayerID() const {return G.size()-1;}
    
    void add2DInput(const int size[3], const bool normalize);
    void addInput(const int size, const bool normalize);
    void addInput(const int size) { addInput(size, false);}
    
    void addConv2DLayer(const int filterSize[3], const int outSize[3], const int padding[2], const int stride[2],
    											const bool normalize, vector<int> linkedTo, const bool bOutput=false);
    void addConv2DLayer(const int filterSize[3], const int outSize[3], const int padding[2], const int stride[2],
    											const bool normalize, const bool bOutput = false) {
    	addConv2DLayer(filterSize, outSize, padding, stride, normalize, vector<int>(), bOutput); }

    void addLayer(const int size, const string type, const bool normalize, vector<int> linkedTo, const bool output);
    void addLayer(const int size, const string type, vector<int> linkedTo) {
        addLayer(size,type,__WHITEN_DEFAULT,linkedTo,false);}
    void addLayer(const int size, const string type, const bool normalize) {
        addLayer(size,type,normalize,vector<int>(),false);}
    void addLayer(const int size, const string type) {
        addLayer(size,type,__WHITEN_DEFAULT,vector<int>(),false);}
    
    void addOutput(const int size, const string type, vector<int> linkedTo) {
        addLayer(size,type,false,linkedTo,true);}
    void addOutput(const int size, const string type) {
        addLayer(size,type,false,vector<int>(),true);}

    Network(const Settings & settings);
    
    ~Network()
    {
        for (auto & trash : G) _dispose_object( trash);
        for (auto & trash : layers) _dispose_object( trash);
        for (auto & trash : mem) _dispose_object( trash);
        for (auto & trash : Vgrad) _dispose_object( trash);
        _dispose_object( grad);
        //_dispose_object(_grad);
        _myfree( weights )
        _myfree( biases )
        _myfree( tgt_weights )
        _myfree( tgt_biases )
        _myfree( weights_DropoutBackup )
    }
    
    void updateFrozenWeights();
    void moveFrozenWeights(const Real alpha);
    void loadMemory(Mem * _M, Activation * _N) const;
    void assignDropoutMask();
    void removeDropoutMask();
    void clearErrors(vector<Activation*>& timeSeries) const;
    void setOutputDeltas(vector<Real>& _errors, Activation* const _N) const;

    Activation* allocateActivation() const;
    vector<Activation*> allocateUnrolledActivations(int length) const;
    void deallocateUnrolledActivations(vector<Activation*>* const ret) const;
    void appendUnrolledActivations(vector<Activation*>* const ret, int length=1) const;

    void predict(const vector<Real>& _input, vector<Real>& _output,
    			vector<Activation*>& timeSeries, const int n_step,
				const Real* const _weights, const Real* const _biases, const Real noise=0.) const;
    void predict(const vector<Real>& _input, vector<Real>& _output,
    			vector<Activation*>& timeSeries, const int n_step, const Real noise=0.) const
    {
        predict(_input, _output, timeSeries, n_step, weights, biases, noise);
    }
    
    void predict(const vector<Real>& _input, vector<Real>& _output,
				Activation* const prevActivation, Activation* const currActivation,
				const Real* const _weights, const Real* const _biases, const Real noise=0.) const;
    void predict(const vector<Real>& _input, vector<Real>& _output,
				Activation* const prevActivation, Activation* const currActivation, const Real noise=0.) const
    {
        predict(_input, _output, prevActivation, currActivation, weights, biases, noise);
    }
    
    void predict(const vector<Real>& _input, vector<Real>& _output, Activation* const net,
    			const Real* const _weights, const Real* const _biases, const Real noise=0.) const;
    void predict(const vector<Real>& _input, vector<Real>& _output, Activation* const net, const Real noise=0.) const
    {
        predict(_input, _output, net, weights, biases, noise);
    }

    void backProp(vector<Activation*>& timeSeries,
    				const Real* const _weights, const Real* const biases, Grads* const _grads) const;
    void backProp(vector<Activation*>& timeSeries, Grads* const _grads) const
    {
    	backProp(timeSeries, weights, biases, _grads);
    }

    void backProp(const vector<Real>& _errors, Activation* const net,
    				const Real* const _weights, const Real* const biases, Grads* const _grads) const;
    void backProp(const vector<Real>& _errors, Activation* const net, Grads* const _grads) const
    {
    	backProp(_errors, net, weights, biases, _grads);
    }


	virtual void printRunning(int counter, std::ostringstream & oa, std::ostringstream & os) {};

	void updateBatchStatistics(Activation* const act) {
		batch_counter++;
		const int invN = 1./batch_counter;
		for(auto & graph : G)
			for(auto & l : *(graph->links))
				l->updateBatchStatistics(running_std, running_avg, act, invN);
	}

	void applyBatchStatistics() {
		const int invNm1 = 1./(batch_counter-1);
      printf("Applying batch statistics from %d samples\n", batch_counter);
		batch_counter = 0;
		for(auto & graph : G)
			for(auto & l : *(graph->links))
				l->applyBatchStatistics(running_std, running_avg, weights, invNm1);
	}

    void resetRunning() {
    	counter=0;

    	for(auto & graph : G)
    		for(auto & l : *(graph->links))
    			l->resetRunning();
    }

    void printRunning() {
    	counter = std::max(counter,2);
    	ostringstream oa;
    	ostringstream os;
    	for(auto & graph : G)
			for(auto & l : *(graph->links))
				l->printRunning(counter, oa, os);
		oa << '\n';
		os << '\n';

    	ofstream outa("running_avg.dat",std::ofstream::app);
    	ofstream outs("running_std.dat",std::ofstream::app);
		if (!outa.good()) die("Unable to open save into avg file\n");
		if (!outs.good()) die("Unable to open save into std file\n");
		outa << oa.str();
		outs << os.str();
    }

    void updateRunning(Activation* const act) {
    	counter++;
    	for(auto & graph : G)
			for(auto & l : *(graph->links))
				l->updateRunning(act, counter);
    }

    void checkGrads(const vector<vector<Real>>& inputs, int lastn=-1);

    void save(const string fname);
    void dump(const int agentID);
    bool restart(const string fname);
};

