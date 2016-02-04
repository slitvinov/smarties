/*
 *  Network.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 20.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <cmath>
#include <armadillo>

#include "Approximator.h"
#include "../rng.h"
#include "LSTMNet.h"
using namespace std;

//namespace ANN
//{
class Layer;
class Link;
class Neuron;

class Network : public Approximator
{
protected:
    
    int nInputs, nOutputs, nLayers;
    
    Real eta;
    Real alpha;
    Real lambda;
    RNG rng;
    int batchSize, nInBatch;
    int totWeights;
    
    vector<Layer*>  layers;
    vector<Real*> inputs;
    vector<Real*> outputs;
    vector<Real*> errors;
    
    arma::mat J;
    arma::mat JtJ;
    arma::mat tmp;
    arma::mat I;
    
    arma::vec e;
    arma::vec dw;
    arma::vec Je;
    
public:
    Network(vector<int>& layerSize, Real eta, Real alpha,Real lambda = 0, int batchSize = -1);
    void predict(const vector<Real>& inputs, vector<Real>& outputs, int nAgent = 0);
    void improve(const vector<Real>& inputs, const vector<Real>& errors, int nAgent = 0);
    void save(string fname) {;}
    bool restart(string fname) { return false; }
    void setBatchsize(int size);
};

class NetworkLM : public Network
{
private:
    Real mu, muFactor, muMin, muMax;
    
    Real Q;
    
    vector< vector<Real> > batch;
    vector< vector<Real> > batchOut;
    vector< vector<Real> > batchExact;
    
public:
    
    NetworkLM(vector<int>& layerSize, Real muFactor, int batchSize = -1);
    void improve(const vector<Real>& inputs, const vector<Real>& errors, int nAgent = 0);
    inline void   rollback();
    inline Real getQ()      { return Q; }
    inline bool   isUpdated() { return nInBatch == 0; }
    
    void save(string fname);
    bool restart(string fname);
    
};


class Link
{
public:
    Neuron* neuronTo;
    Neuron* neuronFrom;
    
    Real  w;
    Real  Dw;
    Real  val;
    Real  err;
    Real  prevDw;
};

class Neuron
{
public:
    ActivationFunction* func;

    vector<Link*>  inLinks;
    vector<Link*>  outLinks;
    
    Real ival, err, oval;

    bool hasInputs, hasOutputs;
    
    Neuron(ActivationFunction* func);		
    void exec();
    void backExec();
    void adjust(Real eta, Real alpha, Real lambda=0);
};

class Layer
{
    public:
    int nNeurons;
    vector<Neuron*> neurons;
    
    Layer(int nNeurons, ActivationFunction* func);
    void propagate();
    void backPropagate();
    void connect(Layer* next, RNG* rng);

    void connect2inputs(vector<Real*>& vals);
    void connect2outputs(vector<Real*>& vals);
    void connect2errors(vector<Real*>& errs);
    void adjust(Real eta, Real alpha, Real lambda);
};

