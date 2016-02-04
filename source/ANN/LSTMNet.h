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
#include "../Settings.h"
using namespace std;

//namespace ANN
//{
	class Layer;
    class HiddenLayer;
	class Link;
	class NeuronOld;
    class Neuron
    class MemoryCell;
    class MemoryBlock;
	class ActivationFunction;

class FishNet: public Approximator
{
protected:
    const vt eta, alpha, lambda, kappa, AdFac;
    const int nAgents;
    vt beta_t_1, beta_t_2, beta_1; beta_2, epsilon;
    int nInputs, nOutputs, nLayers, nNeurons, nWeights, nBiases, ndScelldW, nGates, nStates;
    
    vt *oldvals, *outvals, *in_vals, *errvals, *weights, *biases, *dsdw, *Dw, *Db, *igates, *ogates, *ostates, *nstates;
    /* ADAM optimizer */
    vt *_1stMomW, *_1stMomB, *_2ndMomW, *_2ndMomB;
    vector<NormalLayer*> layers;
    
public:
    
    vt TotSumWeights() {;};
    vt AvgLearnRate()  {;};
    FishNet(vector<int>& layerSize, vector<int>& recurSize, Settings settings, int nAgents);
    void predict(const vector<vt>& input, vector<vt>& output, int nAgent = 0);
    void predict(const vector<vt>& input, vector<vt>& output, int nAgent = 0);
    void improve(const vector<vt>& input, const vector<vt>& error, int nAgent = 0);
    void setBatchsize(int size) {;}
    void save(string fname);
    bool restart(string fname);
};

class NormalLayer
{
    const ActivationFunction * func;
    const int nNeurons, nLinks, n1stNeuron, n1stLink, n1stWeight, n1stBias;
    
    public:
    NormalLayer::NormalLayer(int layerSize, int normal_1st_link, int normal_n_links, int normal_pos, int n1stWeightHL, int n1stBiasHL, ActivationFunction* f) : nNeurons(layerSize), n1stLink(normal_1st_link), nLinks(normal_n_links), n1stNeuron(normal_pos), n1stWeight(n1stWeightHL), n1stBias(n1stBiasHL), func(f)
    {  }
    
    virtual void propagate(vt* in_vals, vt* outvals, vt* oldvals, vt* weights, vt* biases, vt* igates, vt* ogates, vt* ostates, vt* nstates);
    virtual void backPropagate(vt* in_vals, vt* outvals, vt* oldvals, vt* errvals, vt* weights, vt* igates, vt* ogates, vt* ostates, vt* nstates, vt* dsdw, vt* Dw, vt* Db);
};

class LSTMLayer public: NormalLayer
{
    const ActivationFunction * sigm;
    const int nOldL, nIG, nFG, nOG, n1stOld, n1stIG, n1stFG, n1stIN, n1stOG, n1stState, n1stPeep, n1stIGB, n1stFGB, n1stOGB, n1dsdB;
    
    public:
    LSTMLayer::LSTMLayer(int recurSize, int recurr_1st_new_link, int recurr_n_new_links, int recurr_1st_old_link, int recurr_n_old_links, int recurr_tot_links, int recurr_pos, int indIG, int indFG, int indOG, int indState, int n1stWeightIG, int n1stWeightFG, int n1stWeightIN, int n1stWeightOG, int n1stWeightPeep, int n1stBiasIG, int n1stBiasFG, int n1stBiasIN, int n1stBiasOG, int n1stdSdWBias, ActivationFunction* f) :
    NormalLayer(recurSize, recurr_1st_new_link, recurr_n_new_links, recurr_pos, n1stWeightIN, n1stBiasIN, f),
    n1stOld(recurr_1st_old_link), nOldL(recurr_n_old_links), nIG(indIG), nFG(indFG), nOG(indOG), n1stState(indState), n1stIG(n1stWeightIG), n1stFG(n1stWeightFG), n1stOG(n1stWeightOG), n1stPeep(n1stWeightPeep), n1stIGB(n1stBiasIG), n1stFGB(n1stBiasFG), n1stOGB(n1stBiasOG), n1dsdB(n1stdSdWBias), sigm(new SoftSigm)
    { }
    
    void propagate(vt* in_vals, vt* outvals, vt* oldvals, vt* weights, vt* biases, vt* igates, vt* ogates, vt* ostates, vt* nstates);
    void backPropagate(vt* in_vals, vt* outvals, vt* oldvals, vt* errvals, vt* weights, vt* igates, vt* ogates, vt* ostates, vt* nstates, vt* dsdw, vt* Dw, vt* Db);
};

class ActivationFunction
{
public:
    virtual vt eval(vt& arg) = 0;
    virtual vt evalDiff(vt& arg) = 0;
};
class Tanh : public ActivationFunction
{
public:
    inline vt eval(vt& arg)
    {
        if (arg > 20)  return 1;
        if (arg < -20) return -1;
        vt e2x = exp(2.*x);
        return (e2x - 1.) / (e2x + 1.);
    }
    
    inline vt evalDiff(vt& arg)
    {
        if (arg > 20 || arg < -20) return 0;
        
        vt e2x = exp(2.*arg);
        vt t = (e2x + 1.);
        return 4*e2x/(t*t);
    }
};
class Tanh2 : public ActivationFunction
{
public:
    inline vt eval(vt& arg)
    {
        if (arg > 20)  return 2;
        if (arg < -20) return -2;
        vt e2x = exp(2.*x);
        return 2.*(e2x - 1.) / (e2x + 1.);
    }
    
    inline vt evalDiff(vt& arg)
    {
        if (arg > 20 || arg < -20) return 0;
        
        vt e2x = exp(2.*arg);
        vt t = (e2x + 1.);
        return 8.*e2x/(t*t);
    }
};
class Sigm : public ActivationFunction
{
public:
    inline vt eval(vt& arg)
    {
        if (arg > 20)  return 1;
        if (arg < -20) return 0;
        
        vt e_x = exp(-arg);
        return 1. / (1. + e_x);
    }
    
    inline vt evalDiff(vt& arg)
    {
        if (arg > 20 || arg < -20) return 0;
        
        vt ex = exp(arg);
        vt e2x = (1. + ex)*(1. + ex);
        
        return ex/e2x;
    }
};
class Linear : public ActivationFunction
{
public:
    inline vt eval(vt& arg)
    {
        return arg;
    }
    
    inline vt evalDiff(vt& arg)
    {
        return 1;
    }
};
class Gaussian : public ActivationFunction
{
public:
    inline vt eval(vt& x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        if (x > 5 || x < -5) return 0;
        return exp(-10.*x*x);
    }
    inline vt evalDiff(vt& x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        if (x > 5 || x < -5) return 0;
        return -20. * x * exp(-10.*x*x);
    }
};
class SoftSign : public ActivationFunction
{
public:
    inline vt eval(vt& x)
    {
        return x/(1. + fabs(x));
    }
    inline vt evalDiff(vt& x)
    {
        vt denom = 1. + fabs(x);
        return 1./(denom*denom);
    }
};
class SoftSigm : public ActivationFunction
{
public:
    inline vt eval(vt& x)
    {
        return 0.5*(1. + x/(1. + fabs(x)));
    }
    inline vt evalDiff(vt& x)
    {
        vt denom = 1. + fabs(x);
        return 0.5/(denom*denom);
    }
};

