/*
 *  WaveletNet.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 07.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <cmath>
#include <armadillo>

#include "../rng.h"
#include "../ErrorHandling.h"
#include "Approximator.h"

using namespace std;
using namespace ErrorHandling;

class GaussDer
{
public:
    inline Real eval(Real x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        return x * exp(-0.5 * x*x);
    }
    inline Real evalDiff(Real x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        Real x2 = x*x;
        return (1 - x2) * exp(-0.5 * x2);
    }
};

class MexicanHat
{
public:
    inline Real eval(Real x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        Real x2 = x*x;
        return (1 - x2) * exp(-0.5 * x2);
    }
    inline Real evalDiff(Real x)
    {
        if (std::isnan(x) || std::isinf(x)) return 0;
        Real x2 = x*x;
        return x*(x2 - 3) * exp(-0.5 * x2);
    }
};

template<typename Wavelet>
class Wavelon
{
public:
    int dimension;
    vector<Real> m;  // translations
    vector<Real> d;  // dilations
    vector<Real> z;  // scaled individual inputs
    
    vector<Real> frontmul;  // Multiplication of first i wavelets
    vector<Real> backmul;   // Multiplication of last i wavelets
    
    Wavelet wavelet;
    Real outval;
    
    inline Real exec(const vector<Real>& x)
    {
        Real res = 1;
        for (int i=0; i<dimension; i++)
        {
            z[i] = (x[i] - m[i]) / d[i];
            res *= wavelet.eval(z[i]);
            frontmul[i] = res;
        }
        
        res = 1;
        for (int i=dimension-1; i>=0; i--)
        {
            res *= wavelet.eval(z[i]);
            backmul[i] = res;
        }
        
        outval = backmul[0];
        return outval;
    }
    
    inline Real derivate(int k)
    {
        Real front, back;
        
        front = k > 0           ? frontmul[k-1] : 1;
        back  = k < dimension-1 ?  backmul[k+1] : 1;
        
        Real res = front * back * wavelet.evalDiff(z[k]);
        if (std::isnan(res) || std::isinf(res))
            die("NaN error!!\n");
        
        return res;
    }
};


class WaveletNet: public Approximator
{
protected:
    
    int nInputs;
    int nWavelons;
    int nWeights;
    
    Real eta;
    Real alpha;
    RNG rng;
    int batchSize, nInBatch;
    
    vector<Wavelon<MexicanHat>* > wavelons;
    //vector<Real>  inputs;
    Real output;
    Real error;
    
    vector<Real> c;
    vector<Real> a;
    Real a0;
    
    vector<vector<Real> > batch;
    vector<Real> batchOut;
    vector<Real> batchExact;
    
    arma::mat J;
    arma::mat JtJ;
    arma::mat tmp;
    arma::mat I;
    
    arma::vec e;
    arma::vec dw;
    arma::vec w;
    arma::vec prevDw;
    arma::vec Je;
    
    Real mu, muFactor, muMin, muMax;
    
    void computeJ();
    void computeDw();
    void changeWeights();
    void rollback();
    
public:
    
    WaveletNet(vector<int>& layerSize, Real eta, Real alpha, int batchSize = -1);
    void   predict  (const vector<Real>& inputs,       vector<Real>& outputs, int nAgent= 1);
    void   improve  (const vector<Real>& inputs, const vector<Real>& errors, int nAgent= 1);
    void save(string fname);
    bool restart(string fname);
    void setBatchsize(int size);
};


class WaveletNetLM: public WaveletNet
{
    void prepareLM();
    void computeDwLM();
    
public:
    WaveletNetLM(vector<int>& layerSize, int batchSize = -1, Real eta = 1.0) : WaveletNet(layerSize, eta, 1.0, batchSize) {};
    void improve(const vector<Real>& inputs, const vector<Real>& errors, int nAgent= 1);
};