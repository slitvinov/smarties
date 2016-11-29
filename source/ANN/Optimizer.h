/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

//#include <armadillo>
#include "Network.h"
#include "../Profiler.h"

using namespace std;

class Optimizer
{ //basic momentum update
protected:
    const int nWeights, nBiases;
    Network * net;
    Profiler * profiler;
    Real *_1stMomW, *_1stMomB;
    
    void init(Real* const dest, const int N, const Real ini=0);
    void update(Real* const dest, Real* const grad, Real* const _1stMom, const int N, const int batchsize, const Real _lambda=0) const;
    
public:
    const Real eta, lambda, alpha;
    long unsigned nepoch;
    
    Optimizer(Network * _net, Profiler * _prof, Settings & settings);
    
    ~Optimizer()
    {
        _myfree(_1stMomW);
        _myfree(_1stMomB);
    }
    virtual void update(Grads* const G, const int batchsize);
    
    virtual void stackGrads(Grads* const G, const Grads* const g) const;
    virtual void stackGrads(Grads* const G, const vector<Grads*> g) const;
    virtual void stackGrads(const int thrID, Grads* const G, const vector<Grads*> g) const;
};

class AdamOptimizer: public Optimizer
{ //Adam optimizer
protected:
    const Real beta_1, beta_2, epsilon;
    Real beta_t_1, beta_t_2;
    Real *_2ndMomW, *_2ndMomB;
    
    void update(Real* const dest, Real* const grad, Real* const _1stMom, Real* const _2ndMom, const int N, const int batchsize, const Real _lambda=0);
    
public:
    AdamOptimizer(Network * _net, Profiler * _prof, Settings  & settings);
    
    ~AdamOptimizer()
    {
        _myfree(_2ndMomW);
        _myfree(_2ndMomB);
    }
    void update(Grads* const G, const int batchsize) override;
};

/*
class LMOptimizer: public Optimizer
{ //for now just Adam...
    const Real muMax, muMin, muFactor;
    Network * net;
    Profiler * profiler;
    const int nInputs, nOutputs, iOutputs, nWeights, nBiases, totWeights;
    int batchsize;
    Real mu;
    
    arma::mat J;
    arma::mat JtJ;
    arma::mat diagJtJ;
    arma::mat tmp;
    
    arma::vec e;
    arma::vec dw;
    arma::vec Je;
    
    void stackGrads(Grads * g, const int k, const int i);
    void tryNew();
    void goBack();
public:
    LMOptimizer(Network * _net, Profiler * _prof, Settings  & settings);
    
    void trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE);
};
*/
