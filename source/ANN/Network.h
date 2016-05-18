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
    vector<Graph*> G;
    vector<NormalLayer*> layers;
    void orthogonalize(int nO, int nI, int n0);
    void addNormal(Graph * p, Graph * g, bool first, bool last);
    void addLSTM(Graph * p, Graph * g, bool first, bool last);
    //void addInput(Graph & g, int normalSize, int recurrSize);
    void initializeWeights(Graph & g, mt19937 & gen);
    
    //void predict(const vector<Real>& _input, vector<Real>& _output, Lab * _M, Lab * _N, Real * _weights, Real * _biases);
public:
    int nInputs, nOutputs, nLayers, nNeurons, nWeights, nBiases, ndSdW, ndSdB, nStates, iOutputs;
    vector<int> dump_ID;
    vector<Mem*> mem;
    vector<Lab*> series;
    Dsdw * dsdw;
    Grads * grad;///, * _grad;
    bool allocatedFrozenWeights, bDump;
    Real *weights, *biases, *frozen_weights, *frozen_biases;
    
    Network(vector<int>& layerSize, vector<int>& recurSize, Settings & settings);
    
    void allocateSeries(int k, vector<Lab*> & _series);
    void clearDsdw(Dsdw * _dsdw);
    void allocateSeries(int k)
    {
        return allocateSeries(k, series);
    }
    void clearDsdw()
    {
        return clearDsdw(dsdw);
    }
    void clearInputs(Lab* _N);
    void updateFrozenWeights();
    void clearErrors(Lab* _N);
    void clearMemory(Real * outvals, Real * ostates);
    void expandMemory(Mem * _M, Lab * _N);
    void advance(Mem * _M, Lab * _N);
    
    void computeGrads(const vector<Real>& error, Lab * _M, Lab * _N, Grads * _grad);
    void predict(const vector<Real>& input, vector<Real>& output, Lab * _M, Lab * _N, Real * _weights, Real * _biases);
    void predict(const vector<Real>& _input, vector<Real>& _output, Lab * _M, Lab * _N)
    {
        return predict(_input, _output, _M, _N, weights, biases);
    }
    void computeDeltasEnd(vector<Lab*>& _series, const int k);
    void computeDeltasSeries(vector<Lab*>& _series, const int k);
    void computeGradsSeries(vector<Lab*>& _series, const int k, Grads * _grad);
    void computeGradsLightSeries(vector<Lab*>& _series, const int k, Grads * _grad);

    void save(string fname);
    void dump(const int agentID);
    bool restart(string fname);
};

