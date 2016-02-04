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
	inline vt eval(vt x)
	{
		if (std::isnan(x) || std::isinf(x)) return 0;
		return x * exp(-0.5 * x*x);
	}
	inline vt evalDiff(vt x)
	{
		if (std::isnan(x) || std::isinf(x)) return 0;
		vt x2 = x*x;
		return (1 - x2) * exp(-0.5 * x2);
	}
};

class MexicanHat
{
public:
	inline vt eval(vt x)
	{
		if (std::isnan(x) || std::isinf(x)) return 0;
		vt x2 = x*x;
		return (1 - x2) * exp(-0.5 * x2);
	}
	inline vt evalDiff(vt x)
	{
		if (std::isnan(x) || std::isinf(x)) return 0;
		vt x2 = x*x;
		return x*(x2 - 3) * exp(-0.5 * x2);
	}
};

template<typename Wavelet>
class Wavelon
{
public:
	int dimension;
	vector<vt> m;  // translations
	vector<vt> d;  // dilations
	vector<vt> z;  // scaled individual inputs
	
	vector<vt> frontmul;  // Multiplication of first i wavelets
	vector<vt> backmul;   // Multiplication of last i wavelets
	
	Wavelet wavelet;
	vt outval;
	
	inline vt exec(const vector<vt>& x)
	{
		vt res = 1;
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
	
	inline vt derivate(int k)
	{
		vt front, back;
		
		front = k > 0           ? frontmul[k-1] : 1;
		back  = k < dimension-1 ?  backmul[k+1] : 1;
		
		vt res = front * back * wavelet.evalDiff(z[k]);
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
	
	vt eta;
	vt alpha;
	RNG rng;
    int batchSize, nInBatch;
	
	vector<Wavelon<MexicanHat>* > wavelons;
	//vector<vt>  inputs;
	vt output;
	vt error;
	
	vector<vt> c;
	vector<vt> a;
	vt a0;
		
	vector<vector<vt> > batch;
	vector<vt> batchOut;
	vector<vt> batchExact;
	
	arma::mat J;
	arma::mat JtJ;
	arma::mat tmp;
	arma::mat I;
	
	arma::vec e;
	arma::vec dw;
	arma::vec w;
	arma::vec prevDw;
	arma::vec Je;
	
	vt mu, muFactor, muMin, muMax;
	
	void computeJ();
	void computeDw();
	void changeWeights();
	void rollback();
	
public:
    
	WaveletNet(vector<int>& layerSize, vt eta, vt alpha, int batchSize = -1);
	void   predict  (const vector<vt>& inputs,       vector<vt>& outputs);
	void   improve  (const vector<vt>& inputs, const vector<vt>& errors);

	void save(string fname);
	bool restart(string fname);
    void setBatchsize(int size);
};


class WaveletNetLM: public WaveletNet
{
	void prepareLM();
	void computeDwLM();

public:
	WaveletNetLM(vector<int>& layerSize, int batchSize = -1, vt eta = 1.0) : WaveletNet(layerSize, eta, 1.0, batchSize) {};
	void improve(const vector<vt>& inputs, const vector<vt>& errors);	
};





