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
using namespace std;

//namespace ANN
//{
class NormalLayer;
class LSTMLayer;
class ActivationFunction;

class FishNet: public Approximator
{
protected:
    const Real eta, alpha, lambda, kappa, AdFac;
    const int nAgents;
    Real beta_t_1, beta_t_2, beta_1, beta_2, epsilon;
    int nInputs, nOutputs, nLayers, nNeurons, nWeights, nBiases, ndScelldW, nGates, nStates;
    
    Real *oldvals, *outvals, *in_vals, *errvals, *weights, *biases, *dsdw, *Dw, *Db, *igates, *ogates, *ostates, *nstates;
    /* ADAM optimizer */
    Real *_1stMomW, *_1stMomB, *_2ndMomW, *_2ndMomB;
    vector<NormalLayer*> layers;
    
public:
    
    Real TotSumWeights() {;};
    Real AvgLearnRate()  {;};
    FishNet(vector<int>& layerSize, vector<int>& recurSize, Settings settings, int nAgents);
    void predict(const vector<Real>& input, vector<Real>& output, int nAgent = 0);
    void test(const vector<Real>& input, vector<Real>& output, int nAgent = 0);
    void improve(const vector<Real>& input, const vector<Real>& error, int nAgent = 0);
    void setBatchsize(int size) {;}
    void save(string fname);
    bool restart(string fname);
};

class NormalLayer
{
public:
    ActivationFunction * func;
    const int nNeurons, nLinks, n1stNeuron, n1stLink, n1stWeight, n1stBias;

    NormalLayer(int layerSize, int normal_1st_link, int normal_n_links, int normal_pos, int n1stWeightHL, int n1stBiasHL, ActivationFunction* f);
    
    virtual void propagate(Real* in_vals, Real* outvals, Real* oldvals, Real* weights, Real* biases, Real* igates, Real* ogates, Real* ostates, Real* nstates);
    virtual void backPropagate(Real* in_vals, Real* outvals, Real* oldvals, Real* errvals, Real* weights, Real* igates, Real* ogates, Real* ostates, Real* nstates, Real* dsdw, Real* Dw, Real* Db);
};

class LSTMLayer: public NormalLayer
{
public:
    ActivationFunction * sigm;
    const int nOldL, nIG, nFG, nOG, n1stOld, n1stIG, n1stFG, n1stOG, n1stState, n1stPeep, n1stIGB, n1stFGB, n1stOGB, n1dsdB, n1dsIG, n1dsFG, n1dsIN;
    
    LSTMLayer(int recurSize, int recurr_1st_new_link, int recurr_n_new_links, int recurr_1st_old_link, int recurr_n_old_links, int recurr_tot_links, int recurr_pos, int indIG, int indFG, int indOG, int indState, int n1stWeightIG, int n1stWeightFG, int n1stWeightIN, int n1stWeightOG, int n1stWeightPeep, int n1stBiasIG, int n1stBiasFG, int n1stBiasIN, int n1stBiasOG, int n1stdSdWBias, int n1stdSdWIG, int n1stdSdWFG, int n1stdSdWIN, ActivationFunction* f);
    
    void propagate(Real* in_vals, Real* outvals, Real* oldvals, Real* weights, Real* biases, Real* igates, Real* ogates, Real* ostates, Real* nstates);
    void backPropagate(Real* in_vals, Real* outvals, Real* oldvals, Real* errvals, Real* weights, Real* igates, Real* ogates, Real* ostates, Real* nstates, Real* dsdw, Real* Dw, Real* Db);
};

class ActivationFunction
{
public:
    virtual Real eval(const Real& arg) { return 0.;};
    virtual Real evalDiff(const Real& arg) { return 0.;};
    virtual Real eval(const Real * arg) { return eval(*(arg));};
    virtual Real evalDiff(const Real * arg) { return evalDiff(*(arg));};
};
class Tanh : public ActivationFunction
{
public:
    inline Real eval(const Real& arg)
    {
        if (arg > 20)  return 1;
        if (arg < -20) return -1;
        Real e2x = exp(2.*arg);
        return (e2x - 1.) / (e2x + 1.);
    }
    
    inline Real evalDiff(const Real& arg)
    {
        if (arg > 20 || arg < -20) return 0;
        
        Real e2x = exp(2.*arg);
        Real t = (e2x + 1.);
        return 4*e2x/(t*t);
    }
};
class Tanh2 : public ActivationFunction
{
public:
    inline Real eval(const Real& arg)
    {
        if (arg > 20)  return 2;
        if (arg < -20) return -2;
        Real e2x = exp(2.*arg);
        return 2.*(e2x - 1.) / (e2x + 1.);
    }
    
    inline Real evalDiff(const Real& arg)
    {
        if (arg > 20 || arg < -20) return 0;
        
        Real e2x = exp(2.*arg);
        Real t = (e2x + 1.);
        return 8.*e2x/(t*t);
    }
};
class Sigm : public ActivationFunction
{
public:
    inline Real eval(const Real& arg)
    {
        if (arg > 20)  return 1;
        if (arg < -20) return 0;
        
        Real e_x = exp(-arg);
        return 1. / (1. + e_x);
    }
    
    inline Real evalDiff(const Real& arg)
    {
        if (arg > 20 || arg < -20) return 0;
        
        Real ex = exp(arg);
        Real e2x = (1. + ex)*(1. + ex);
        
        return ex/e2x;
    }
};
class Linear : public ActivationFunction
{
public:
    inline Real eval(const Real& arg)
    {
        return arg;
    }
    
    inline Real evalDiff(const Real& arg)
    {
        return 1;
    }
};
class Gaussian : public ActivationFunction
{
public:
    inline Real eval(const Real& x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        if (x > 5 || x < -5) return 0;
        return exp(-10.*x*x);
    }
    inline Real evalDiff(const Real& x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        if (x > 5 || x < -5) return 0;
        return -20. * x * exp(-10.*x*x);
    }
};
class SoftSign : public ActivationFunction
{
public:
    inline Real eval(const Real& x)
    {
        return x/(1. + fabs(x));
    }
    inline Real evalDiff(const Real& x)
    {
        Real denom = 1. + fabs(x);
        return 1./(denom*denom);
    }
};
class SoftSigm : public ActivationFunction
{
public:
    inline Real eval(const Real& x)
    {
        return 0.5*(1. + x/(1. + fabs(x)));
    }
    inline Real evalDiff(const Real& x)
    {
        Real denom = 1. + fabs(x);
        return 0.5/(denom*denom);
    }
};

