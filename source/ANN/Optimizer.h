/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <random>
//#include <armadillo>
#include "Network.h"
#include "../Profiler.h"

using namespace std;

class Optimizer
{ //for now just Adam...
public:
    virtual void update(Grads * G) {};
    virtual void addUpdate(Grads * G) {};
    virtual void stackGrads(Grads * G, Grads * g) {};
    virtual void trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) {die("Wrong algo, dude\n");};
    virtual void trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE) {die("Wrong algo, dude\n");};
    
    virtual void trainQ(const vector<vector<Real>> & states, const vector<int> & actions, const vector<Real> & rewards, function<void(vector<Real>, int, Real, vector<Real>)> & errs, const int iAgent) {die("Wrong algo, dude\n");};
    virtual void trainDQ(const vector<vector<Real>> & states, const vector<int> & actions, const vector<Real> & rewards, function<void(vector<Real>, int, Real, vector<Real>, vector<Real>)> & errs, const int iAgent) {die("Wrong algo, dude\n");};
    virtual void trainSeries2(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) {die("Wrong algo, dude\n");};
    virtual void trainSeries3(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) {die("Wrong algo, dude\n");};
    virtual void trainSeries4(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) {die("Wrong algo, dude\n");};
    virtual void trainSeries5(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) {die("Wrong algo, dude\n");};
    virtual void checkGrads(const vector<vector<Real>>& inputs) {die("Wrong algo, dude\n");};
};

class AdamOptimizer: public Optimizer
{ //for now just Adam...
    const Real eta, beta_1, beta_2, epsilon;
    Network * net;
    Profiler * profiler;
    const int nInputs, nOutputs, iOutputs, nWeights, nBiases;
    Real beta_t_1, beta_t_2;
    int batchsize, nepoch;
    Real *_1stMomW, *_1stMomB, *_2ndMomW, *_2ndMomB, *_etaW, *_etaB;
    
    void init(Real* dest, const int N, Real ini=0);
    void update(Real* dest, Real* grad, Real* _1stMom, Real* _2ndMom, const int N, Real _eta);
    void update(Real* dest, Real* grad, const int N, Real _eta);
    void update(Real* dest, Real* grad, Real* _1stMom, Real* _2ndMom, Real* _eta, const int N);
    void stackGrads(Real * G, Real * g, Real* _1stMom, Real* _2ndMom, const int N);
public:
    AdamOptimizer(Network * _net, Profiler * _prof, Settings  & settings);
    
    void update(Grads * G) override;
    void addUpdate(Grads * G) override;
    void stackGrads(Grads * G, Grads * g) override;
    void checkGrads(const vector<vector<Real>>& inputs) override;
    void trainSeries(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) override;
    void trainSeries2(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) override;
    void trainSeries3(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) override;
    void trainSeries4(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) override;
    void trainSeries5(const vector<vector<Real>>& inputs, const vector<vector<Real>>& targets, Real & trainMSE) override;
    void trainBatch(const vector<const vector<Real>*>& inputs, const vector<const vector<Real>*>& targets, Real & trainMSE) override;
    
    //void trainQ(const vector<vector<Real>> & states, const vector<int> & actions, const vector<Real> & rewards, function<void(vector<Real>, int, Real, vector<Real>)> & errs, const int iAgent);
    //void trainDQ(const vector<vector<Real>> & states, const vector<int> & actions, const vector<Real> & rewards, function<void(vector<Real>, int, Real, vector<Real>, vector<Real>)> & errs, const int iAgent);
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