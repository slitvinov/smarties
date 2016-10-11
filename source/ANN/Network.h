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
    int nAgents, nThreads, nInputs, nOutputs, nLayers, nNeurons, nWeights, nBiases, nStates;
    const Real Pdrop; //dropout
    vector<int> iOut, dump_ID;
    vector<Graph*> G;
    vector<NormalLayer*> layers;
    void build_LSTM_layer(Graph* const graph);
    void build_normal_layer(Graph* const graph);

public:
    bool allocatedFrozenWeights, allocatedDroputWeights, backedUp, bDump;
    mt19937 * gen;
    vector<Mem*> mem;
    vector<Activation*> series;
    Grads * grad, * _grad;
    Real *weights, *biases, *tgt_weights, *tgt_biases, *weights_DropoutBackup;
    vector<Grads*> Vgrad;

    void build();
    void addInput(const int size);
    int getnWeights() const {assert(bBuilt); return nWeights;}
    int getnBiases() const {assert(bBuilt); return nBiases;}
    int getnOutputs() const {assert(bBuilt); return nOutputs;}
    int getnInputs() const {assert(bBuilt); return nInputs;}
    int getLastLayerID() const {return G.size()-1;}
    void addLayer(const int size, const string type, vector<int> linkedTo, const bool output);
    void addLayer(const int size, const string type, vector<int> linkedTo) {addLayer(size,type,linkedTo,false);}
    void addLayer(const int size, const string type, const bool output) {addLayer(size,type,vector<int>(),output);}
    void addLayer(const int size, const string type) {addLayer(size,type,vector<int>(),false);}
    void addOutput(const int size, const string type, vector<int> linkedTo) {addLayer(size,type,linkedTo,true);}
    void addOutput(const int size, const string type) {addLayer(size,type,vector<int>(),true);}

    Network(const Settings & settings);
    
    ~Network()
    {
        for (auto & trash : G) _dispose_object( trash);
        for (auto & trash : layers) _dispose_object( trash);
        for (auto & trash : mem) _dispose_object( trash);
        for (auto & trash : series) _dispose_object( trash);
        for (auto & trash : Vgrad) _dispose_object( trash);
        _dispose_object( grad);
        _dispose_object(_grad);
        _myfree( weights )
        _myfree( biases )
        _myfree( tgt_weights )
        _myfree( tgt_biases )
        _myfree( weights_DropoutBackup )
    }
    
    void allocateSeries(int k, vector<Activation*> & _series);
    void allocateSeries(int k)
    {
        return allocateSeries(k, series);
    }
    
    void updateFrozenWeights();
    void moveFrozenWeights(const Real alpha);
    void expandMemory(Mem * _M, Activation * _N) const;
    void assignDropoutMask();
    void removeDropoutMask();
    
    void predict(const vector<Real>& _input, vector<Real>& _output, const Activation* const _M, Activation* const _N, const Real* const _weights, const Real* const _biases) const;
    void predict(const vector<Real>& _input, vector<Real>& _output, const Activation* const _M, Activation* const _N) const
    {
        predict(_input, _output, _M, _N, weights, biases);
    }
    
    void predict(const vector<Real>& _input, vector<Real>& _output, Activation* const _N, const Real* const _weights, const Real* const _biases) const;
    void predict(const vector<Real>& _input, vector<Real>& _output, Activation* const _N) const
    {
        predict(_input, _output, _N, weights, biases);
    }
    
    void setOutputErrors(vector<Real>& _errors, Activation* const _N);
    
    void computeDeltas(Activation* const _series, const Real* const _weights, const Real* const _biases) const;
    void computeDeltas(Activation* const _series) const
    {
        computeDeltas(_series, weights, biases);
    }
    
    void computeDeltasInputs(vector<Real>& grad, const Activation* const _series, const Real* const _weights, const Real* const _biases) const;
    void computeDeltasInputs(vector<Real>& grad, const Activation* const _series) const
    {
        computeDeltasInputs(grad, _series, weights, biases);
    }
    
    void computeDeltasSeries(vector<Activation*>& _series, const int first, const int last, const Real* const _weights, const Real* const _biases) const;
    void computeDeltasSeries(vector<Activation*>& _series, const int first, const int last) const
    {
        computeDeltasSeries(_series, first, last, weights, biases);
    }
    
    void computeGradsSeries(const vector<Activation*>& _series, const int k, Grads* const _Grad, const Real* const _weights) const;
    void computeGradsSeries(const vector<Activation*>& _series, const int k, Grads* const _Grad) const
    {
        computeGradsSeries(_series, k, _Grad, weights);
    }
    void computeGrads(const Activation* const _series, Grads* const _Grad, const Real* const _weights) const;
    void computeGrads(const Activation* const _series, Grads* const _Grad) const
    {
        computeGrads(_series, _Grad, weights);
    }
    
    void computeAddGradsSeries(const vector<Activation*>& _series, const int first, const int last, Grads* const _Grad, const Real* const _weights) const;
    void computeAddGradsSeries(const vector<Activation*>& _series, const int first, const int last, Grads* const _Grad) const
    {
        computeAddGradsSeries(_series, first, last, _Grad, weights);
    }
    void computeAddGrads(const Activation* const _series, Grads* const _Grad, const Real* const _weights) const;
    void computeAddGrads(const Activation* const _series, Grads* const _Grad) const
    {
        computeAddGrads(_series, _Grad, weights);
    }
    
    void checkGrads(const vector<vector<Real>>& inputs, const int lastn);

    void save(const string fname);
    void dump(const int agentID);
    bool restart(const string fname);
};

