/*
 *  LSTMNet.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

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
    Real *weights, *biases, *tgt_weights, *tgt_biases, *weights_DropoutBackup;
    vector<Grads*> Vgrad;

    int counter;
    vector<Real> runningAvg, runningStd;

    void build();
    
    int getnWeights() const {assert(bBuilt); return nWeights;}
    int getnBiases() const {assert(bBuilt); return nBiases;}
    int getnOutputs() const {assert(bBuilt); return nOutputs;}
    int getnInputs() const {assert(bBuilt); return nInputs;}
    int getLastLayerID() const {return G.size()-1;}
    
    void add2DInput(const int size[3], const bool normalize);
    void addInput(const int size, const bool normalize);
    void addInput(const int size) {
        addInput(size, true);}
    
    void addConv2DLayer(const int filterSize[3], const int outSize[3], const int padding[2], const int stride[2],
    											const bool normalize, vector<int> linkedTo, const bool bOutput=false);
    void addConv2DLayer(const int filterSize[3], const int outSize[3], const int padding[2], const int stride[2],
    											const bool normalize, const bool bOutput = false) {
    	addConv2DLayer(filterSize, outSize, padding, stride, normalize, vector<int>(), bOutput); }

    void addLayer(const int size, const string type, const bool normalize, vector<int> linkedTo, const bool output);
    void addLayer(const int size, const string type, vector<int> linkedTo) {
        addLayer(size,type,true,linkedTo,false);}
    void addLayer(const int size, const string type, const bool normalize) {
        addLayer(size,type,normalize,vector<int>(),false);}
    void addLayer(const int size, const string type) {
        addLayer(size,type,true,vector<int>(),false);}
    
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
    				const Real* const _weights, Grads* const _grads) const;
    void backProp(vector<Activation*>& timeSeries, Grads* const _grads) const
    {
    	backProp(timeSeries, weights, _grads);
    }

    void backProp(const vector<Real>& _errors, Activation* const net,
    				const Real* const _weights, Grads* const _grads) const;
    void backProp(const vector<Real>& _errors, Activation* const net, Grads* const _grads) const
    {
    	backProp(_errors, net, weights, _grads);
    }

    void resetRunning() {
    	counter=0;

    	if(runningAvg.size() != nNeurons) runningAvg.resize(nNeurons);
    	if(runningStd.size() != nNeurons) runningStd.resize(nNeurons);

    	for (int k=0; k<nNeurons; k++) {
    		runningAvg[k] = 0;
    		runningStd[k] = 0;
    	}
    }

    void printRunning() {
    	counter = std::max(counter,2);

    	ofstream outa("running_avg.txt",std::ofstream::app);
    	ofstream outs("running_std.txt",std::ofstream::app);
		if (!outa.good()) die("Unable to open save into avg file\n");
		if (!outs.good()) die("Unable to open save into std file\n");
    	const Real invNm1 = 1./(counter-1);

		for (int i=nInputs; i<nNeurons; i++)  outa << runningAvg[i] << " ";
		for (int i=nInputs; i<nNeurons; i++)  outs << runningStd[i]*invNm1 << " ";
		outa << '\n';
		outs << '\n';
    }

    void updateRunning(Activation* const act) {
    	counter++;
    	assert(runningAvg.size() == nNeurons);
    	assert(runningStd.size() == nNeurons);
    	const Real invN = 1./counter;
    	for (int k=nInputs; k<nNeurons; k++) {
    		const Real delta = act->in_vals[k] - runningAvg[k];
    		runningAvg[k] += delta*invN;
    		runningStd[k] += delta*(act->in_vals[k] - runningAvg[k]);
    	}
    }

    void checkGrads(const vector<vector<Real>>& inputs, int lastn=-1);

    void save(const string fname);
    void dump(const int agentID);
    bool restart(const string fname);
};

