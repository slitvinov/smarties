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
{ //for now just Adam...
public:
    //int batchsize;
    int nepoch;
    virtual void update(Grads* const G, const int batchsize) {};
    virtual void addUpdate(Grads* const G) {};
    virtual void stackGrads(Grads* const G, const Grads* const g) const {};
    virtual void stackGrads(Grads* const G, const vector<Grads*> g) {};
    virtual void trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) {die("Wrong algo, dude\n");};
    virtual void trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE) {die("Wrong algo, dude\n");};
    virtual void checkGrads(const vector<vector<Real>>& inputs, const int lastn, const int ierr) {die("Wrong algo, dude\n");};
};

class AdamOptimizer: public Optimizer
{ //for now just Adam...
    const Real eta, beta_1, beta_2, epsilon, lambda;
    Network * net;
    Profiler * profiler;
    const int nInputs, nOutputs, iOutputs, nWeights, nBiases;
    Real beta_t_1, beta_t_2;
    Real *_1stMomW, *_1stMomB, *_2ndMomW, *_2ndMomB;
    
    void init(Real* const dest, const int N, const Real ini=0);
    void update(Real* const dest, Real* const grad, Real* const _1stMom, Real* const _2ndMom, const int N, const Real _eta);
    void updateDecay(Real* const dest, Real* const grad, Real* const _1stMom, Real* const _2ndMom, const int N, const Real _eta);
    void update(Real* const dest, Real* const grad, const int N, const Real _eta) const;
    void stackGrads(Real* const G, const Real* const g, Real* const _1stMom, Real* const _2ndMom, const int N);
public:
    AdamOptimizer(Network * _net, Profiler * _prof, Settings  & settings);
    
    void update(Grads* const G, const int batchsize) override;
    void addUpdate(Grads* const G) override;
    void stackGrads(Grads* const G, const Grads* const g) const override;
    void stackGrads(Grads* const G, const vector<Grads*> g) override;
    void checkGrads(const vector<vector<Real>>& inputs, const int lastn, const int ierr) override;
    void trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) override;
    void trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE) override;
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
