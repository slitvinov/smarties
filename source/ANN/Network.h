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
    void orthogonalize(const int nO, const int nI, const int n0, Real* const _weights);
    void addNormal(Graph* const p, Graph* const g, const bool first, const bool last);
    void addLSTM(Graph* const p, Graph* const g, const bool first, const bool last);
    //void addInput(Graph & g, int normalSize, int recurrSize);
    void initializeWeights(Graph & g, Real* const _weights, Real* const _biases);
    
    //void predict(const vector<Real>& _input, vector<Real>& _output, Lab * _M, Lab * _N, Real * _weights, Real * _biases);
public:
    const Real Pdrop;
    const int nInputs, nOutputs;
    int nLayers, nNeurons, nWeights, nBiases, ndSdW, ndSdB, nStates, iOutputs;
    vector<int> dump_ID;
    bool allocatedFrozenWeights, allocatedDroputWeights, backedUp, bDump;
    mt19937 * gen;
    vector<Mem*> mem;
    vector<Lab*> series;
    Grads * grad, * _grad;
    vector<Grads*> Vgrad;
    Real *weights, *biases, *frozen_weights, *frozen_biases, *weights_DropoutBackup;
    
    Network(const vector<int>& layerSize, const bool bLSTM, const Settings & settings);
    Network(const vector<int>& layerSize, const vector<int>& recurSize, const Settings & settings);
    
    void allocateSeries(int k, vector<Lab*> & _series);
    void allocateSeries(int k)
    {
        return allocateSeries(k, series);
    }
    
    void clearInputs(Lab* _N);
    void updateFrozenWeights();
    void moveFrozenWeights(const Real alpha);
    void clearErrors(Lab* _N) const;
    void clearMemory(Real * outvals, Real * ostates) const;
    void expandMemory(Mem * _M, Lab * _N) const;
    void advance(Mem * _M, Lab * _N);
    void assignDropoutMask();
    void removeDropoutMask();
    
    void computeGrads(const vector<Real>& _error, const Lab* const _M, Lab* const _N, Grads* const _Grad) const;
    
    void predict(const vector<Real>& _input, vector<Real>& _output, const Lab* const _M, Lab* const _N, const Real* const _weights, const Real* const _biases) const;
    void predict(const vector<Real>& _input, vector<Real>& _output, const Lab* const _M, Lab* const _N) const
    {
        predict(_input, _output, _M, _N, weights, biases);
    }
    
    void predict(const vector<Real>& _input, vector<Real>& _output, Lab* const _N, const Real* const _weights, const Real* const _biases) const;
    void predict(const vector<Real>& _input, vector<Real>& _output, Lab* const _N) const
    {
        predict(_input, _output, _N, weights, biases);
    }
    
    void computeDeltas(Lab* const _series) const;
    void computeDeltasInputs(vector<Lab*>& _series, const int k) const;
    void computeDeltasSeries(vector<Lab*>& _series, const int first, const int last) const;
    
    void computeGradsSeries(const vector<Lab*>& _series, const int k, Grads* const _Grad) const;
    void computeGrads(const Lab* const _series, Grads* const _Grad) const;
    void computeAddGradsSeries(const vector<Lab*>& _series, const int k, Grads* const _Grad) const;
    void computeAddGrads(const Lab* const _series, Grads* const _Grad) const;

    void save(const string fname);
    void dump(const int agentID);
    bool restart(const string fname);
};

