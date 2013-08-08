/*
 *  WaveletNet.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 07.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/operations.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>

#include "WaveletNet.h"

WaveletNet::WaveletNet(vector<int>& layerSize, double eta, double alpha, int batchSize) :
eta(eta), alpha(alpha), nInputs(layerSize[0]), nWavelons(layerSize[1]), rng(0), batchSize(batchSize)
{
	nWeights = 1 + nInputs + nWavelons + nInputs*nWavelons * 2;
	a.resize(nInputs);
	c.resize(nWavelons);
	wavelons.resize(nWavelons);
	
	for (int i=0; i<nWavelons; i++)
	{
		wavelons[i] = new Wavelon<GaussDer>();
		wavelons[i]->m.resize(nInputs);
		wavelons[i]->d.resize(nInputs);
		wavelons[i]->z.resize(nInputs);
		wavelons[i]->frontmul.resize(nInputs);
		wavelons[i]->backmul.resize(nInputs);
		
		wavelons[i]->dimension = nInputs;
		for (int j=0; j<nInputs; j++)
		{
			wavelons[i]->m[j] = 0 + rng.uniform(-1, 1);
			wavelons[i]->d[j] = 1.8 + rng.uniform(-0.01, 0.01);
		}
	}
	
	J.resize(batchSize, nWeights);
	I = ublas::identity_matrix<double>(nWeights);
	
	e.resize(batchSize);
	dw.resize(nWeights);
	prevDw.resize(nWeights);
	
	nInBatch = 0;	
	
	for (int i=0; i<nWeights; i++)
		prevDw(i) = 0;
	
	for (int i=0; i<nWavelons; i++)
		c[i] = rng.uniform(-0.01, 0.01);
	for (int i=0; i<nInputs; i++)
		a[i] = rng.uniform(-0.01, 0.01);
	
	a0 = rng.uniform(-0.01, 0.01);
	
	batch.resize(batchSize);
	batchOut.resize(batchSize);
	batchExact.resize(batchSize);
	
	mu = 0.01;
	muFactor = 5;
	muMin = 1e-2;
	muMax = 1e+10;
}

void WaveletNet::predict(const vector<double>& inputs, vector<double>& outputs)
{
	double res = 0;
	
	// Bias
	
	res += a0;
	
	// Linear terms
	
	for (int i=0; i<nInputs; i++)
		res += inputs[i] * a[i];
	
	// Wavelet terms
	
	for (int i=0; i<nWavelons; i++)
		res += wavelons[i]->exec(inputs) * c[i];

	outputs[0] = res;	
}

void WaveletNet::computeJ()
{
	int iw = 0;
	
	// Bias
	
	J(nInBatch, iw++) = 1;
	
	// Linear terms
	
	for (int i=0; i < nInputs; i++)
		J(nInBatch, iw++) = batch[nInBatch][i];
	
	// Summing weights
	
	for (int i=0; i < nWavelons; i++)
		J(nInBatch, iw++) = wavelons[i]->outval;
	
	// Translations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			J(nInBatch, iw++) = - c[j] / wavelons[j]->d[k] * wavelons[j]->derivate(k);
	
	// Dilations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			J(nInBatch, iw++) = - c[j] / wavelons[j]->d[k] * wavelons[j]->z[k] * wavelons[j]->derivate(k);
}

void WaveletNet::computeDw()
{
	dw = -eta * ublas::prod(ublas::trans(J), e) + alpha * prevDw;
	prevDw = -eta * ublas::prod(ublas::trans(J), e);
}

void WaveletNet::computeDwLM()
{
	J = -J;
	JtJ = ublas::prod(ublas::trans(J), J);
	dw  = ublas::prod(ublas::trans(J), e);
	tmp = JtJ + mu*I;
	
	ublas::permutation_matrix<double> piv(tmp.size1()); 
	ublas::lu_factorize(tmp, piv);
	ublas::lu_substitute(tmp, piv, dw);
	
	dw *= eta;
}

void WaveletNet::changeWeights()
{
	// Change the weights
	
	int iw = 0;
	
	// Bias
	
	a0 += dw(iw++);
	
	// Linear terms
	
	for (int i=0; i < nInputs; i++)
		a[i] += dw(iw++);
	
	// Summing weights
	
	for (int i=0; i < nWavelons; i++)
		c[i] += dw(iw++);
	
	// Translations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			wavelons[j]->m[k] += dw(iw++);
	
	// Dilations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			wavelons[j]->d[k] += dw(iw++);	
}

void WaveletNet::rollback()
{
	// Unchange the weights
	
	int iw = 0;
	
	// Bias
	
	a0 -= dw(iw++);
	
	// Linear terms
	
	for (int i=0; i < nInputs; i++)
		a[i] -= dw(iw++);
	
	// Summing weights
	
	for (int i=0; i < nWavelons; i++)
		c[i] -= dw(iw++);
	
	// Translations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			wavelons[j]->m[k] -= dw(iw++);
	
	// Dilations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			wavelons[j]->d[k] -= dw(iw++);	
}	

void WaveletNet::improve(const vector<double>& inputs, const vector<double>& errors)
{
	// Save all the info about current object
	
	vector<double> tmpVec(1);
	//predict(inputs, tmpVec);
	batch[nInBatch] = inputs;
	//batchOut[nInBatch] = tmpVec[0];
	//batchExact[nInBatch] = tmpVec[0] - errors[0];
	e(nInBatch) = errors[0];
	
	// Compute grad for this object
	
	computeJ();
	
	nInBatch++;
	
	// If batch is full we have to update
	
	if (nInBatch == batchSize)
	{
		computeDwLM();
		changeWeights();
		
		nInBatch = 0;
	}		
}

void WaveletNet::improveLM(const vector<double>& inputs, const vector<double>& errors)
{
	// Save all the info about current object
	
	vector<double> tmpVec(1);
	predict(inputs, tmpVec);
	batch[nInBatch] = inputs;
	batchOut[nInBatch] = tmpVec[0];
	batchExact[nInBatch] = tmpVec[0] - errors[0];
	e(nInBatch) = errors[0];
	
	// Compute grad for this object
	
	computeJ();
	
	nInBatch++;
	
	// If batch is full we have to update
	
	if (nInBatch == batchSize)
	{
		double Q, Q0 = 0;
		
		for (int i=0; i<batchSize; i++)
			Q0 += e(i) * e(i);
		
		Q = Q0+1;
		
				
		while (Q > Q0)
		{
			computeDwLM();
			
			changeWeights();
			
			Q = 0;
			for (int i=0; i<batchSize; i++)
			{
				predict(batch[i], tmpVec);
				
				double diff = tmpVec[0] - batchExact[i];
				Q += diff * diff;
			}
			
			if (Q > Q0)
			{
				if (mu < muMax)
					mu *= muFactor;
				else
				{
					break;
				}
				rollback();
			}
		}
		
		if (mu > muMin) mu /= muFactor;
		
		nInBatch = 0;
	}		
}




















