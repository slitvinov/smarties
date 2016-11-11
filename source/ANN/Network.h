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

public:
    int nAgents, nThreads, nInputs, nOutputs, nLayers, nNeurons, nWeights, nBiases, nStates;
    bool allocatedFrozenWeights, allocatedDroputWeights, backedUp, bDump;
    mt19937 * gen;
    vector<Mem*> mem;
    Grads * grad;//, * _grad;
    Real *weights, *biases, *tgt_weights, *tgt_biases, *weights_DropoutBackup;
    vector<Grads*> Vgrad;

    void build();
    
    int getnWeights() const {assert(bBuilt); return nWeights;}
    int getnBiases() const {assert(bBuilt); return nBiases;}
    int getnOutputs() const {assert(bBuilt); return nOutputs;}
    int getnInputs() const {assert(bBuilt); return nInputs;}
    int getLastLayerID() const {return G.size()-1;}
    
    void addInput(const int size, const bool normalize);
    void addInput(const int size) {
        addInput(size, true);}
    
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
    
    void checkGrads(const vector<vector<Real>>& inputs, int lastn=-1);

    void save(const string fname);
    void dump(const int agentID);
    bool restart(const string fname);
};

