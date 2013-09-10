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
	inline double eval(double x)
	{
		if (std::isnan(x) || std::isinf(x)) return 0;
		return x * exp(-0.5 * x*x);
	}
	inline double evalDiff(double x)
	{
		if (std::isnan(x) || std::isinf(x)) return 0;
		double x2 = x*x;
		return (1 - x2) * exp(-0.5 * x2);
	}
};

class MexicanHat
{
public:
	inline double eval(double x)
	{
		if (std::isnan(x) || std::isinf(x)) return 0;
		double x2 = x*x;
		return (1 - x2) * exp(-0.5 * x2);
	}
	inline double evalDiff(double x)
	{
		if (std::isnan(x) || std::isinf(x)) return 0;
		double x2 = x*x;
		return x*(x2 - 3) * exp(-0.5 * x2);
	}
};

template<typename Wavelet>
class Wavelon
{
public:
	int dimension;
	vector<double> m;  // translations
	vector<double> d;  // dilations
	vector<double> z;  // scaled individual inputs
	
	vector<double> frontmul;  // Multiplication of first i wavelets
	vector<double> backmul;   // Multiplication of last i wavelets
	
	Wavelet wavelet;
	double outval;
	
	inline double exec(const vector<double>& x)
	{
		double res = 1;
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
	
	inline double derivate(int k)
	{
		double front, back;
		
		front = k > 0           ? frontmul[k-1] : 1;
		back  = k < dimension-1 ?  backmul[k+1] : 1;
		
		double res = front * back * wavelet.evalDiff(z[k]);
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
	
	double eta;
	double alpha;
	RNG rng;
	
	vector<Wavelon<GaussDer>* > wavelons;
	//vector<double>  inputs;
	double output;
	double error;
	
	vector<double> c;
	vector<double> a;
	double a0;
		
	vector<vector<double> > batch;
	vector<double> batchOut;
	vector<double> batchExact;
	int batchSize;
	int nInBatch;
	
	arma::mat J;
	arma::mat JtJ;
	arma::mat tmp;
	arma::mat I;
	
	arma::vec e;
	arma::vec dw;
	arma::vec w;
	arma::vec prevDw;
	arma::vec Je;
	
	double mu, muFactor, muMin, muMax;
	
	void computeJ();
	void computeDw();
	void changeWeights();
	void rollback();
	
public:
	
	WaveletNet(vector<int>& layerSize, double eta, double alpha, int batchSize = -1);
	void   predict  (const vector<double>& inputs,       vector<double>& outputs);
	void   improve  (const vector<double>& inputs, const vector<double>& errors);
	
	void save(string fname);
	bool restart(string fname);		
};


class WaveletNetLM: public WaveletNet
{
	void prepareLM();
	void computeDwLM();

public:
	WaveletNetLM(vector<int>& layerSize, int batchSize = -1, double eta = 1.0) : WaveletNet(layerSize, eta, 1.0, batchSize) {};
	void improve(const vector<double>& inputs, const vector<double>& errors);	
};





