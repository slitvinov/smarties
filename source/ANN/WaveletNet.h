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
#include <boost/numeric/ublas/matrix.hpp>

#include "../rng.h"
#include "../ErrorHandling.h"

using namespace std;
using namespace ErrorHandling;
namespace ublas = boost::numeric::ublas;

class GaussDer
{
public:
	inline double eval(double x)
	{
		return x * exp(-0.5 * x*x);
	}
	inline double evalDiff(double x)
	{
		double x2 = x*x;
		return (1 - x2) * exp(-0.5 * x2);
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
		
		outval = pow(4, dimension) * backmul[0];
		return outval;
	}
	
	inline double derivate(int k)
	{
		double front, back;
		
		front = k > 0           ? frontmul[k-1] : 1;
		back  = k < dimension-1 ?  backmul[k+1] : 1;
		
		double res = pow(4, dimension) * front * back * wavelet.evalDiff(z[k]);
		if (isnan(res) || isinf(res))
			error("NaN error!!\n");
		
		return res;
	}
};


class WaveletNet
{
private:
	
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
	
	ublas::matrix<double> J;
	ublas::matrix<double> JtJ;
	ublas::matrix<double> tmp;
	ublas::matrix<double> I;
	
	ublas::vector<double> e;
	ublas::vector<double> dw;
	ublas::vector<double> prevDw;
	
	double mu, muFactor, muMin, muMax;
	
	void computeJ();
	void computeDw();
	void changeWeights();
	void rollback();
	void computeDwLM();
		
public:
	
	WaveletNet(vector<int>& layerSize, double eta, double alpha, int batchSize);
	void   predict  (const vector<double>& inputs, vector<double>& outputs);
	void   improve  (const vector<double>& inputs, const vector<double>& errors);		
	void   improveLM(const vector<double>& inputs, const vector<double>& errors);		
};








