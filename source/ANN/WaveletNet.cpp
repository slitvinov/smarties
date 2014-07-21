/*
 *  WaveletNet.cpp
 *  rl
 *
 *  Created by Dmitry Alexeev on 07.08.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include <cmath>

#include "WaveletNet.h"

WaveletNet::WaveletNet(vector<int>& layerSize, double eta, double alpha, int batchSize) :
eta(eta), alpha(alpha), nInputs(layerSize[0]), nWavelons(layerSize[1]), rng(0), batchSize(batchSize)
{
	nWeights = nWavelons + nInputs*nWavelons * 2;
	if (batchSize == -1) this->batchSize = batchSize = nWeights;

	a.resize(nInputs);
	c.resize(nWavelons);
	wavelons.resize(nWavelons);
	
	for (int i=0; i<nWavelons; i++)
	{
		wavelons[i] = new Wavelon<MexicanHat>();
		wavelons[i]->m.resize(nInputs);
		wavelons[i]->d.resize(nInputs);
		wavelons[i]->z.resize(nInputs);
		wavelons[i]->frontmul.resize(nInputs);
		wavelons[i]->backmul.resize(nInputs);
		
		wavelons[i]->dimension = nInputs;
		for (int j=0; j<nInputs; j++)
		{
			wavelons[i]->m[j] = 0.0 + rng.uniform(-0.5, 0.5);
			wavelons[i]->d[j] = 1.2 + rng.uniform(-0.1, 0.1);
		}
	}
	
	J.set_size(batchSize, nWeights);
	I.eye(nWeights, nWeights);
	
	e.set_size(batchSize);
	w.set_size(nWeights);
	dw.set_size(nWeights);
	Je.set_size(nWeights);
	prevDw.set_size(nWeights);
	
	nInBatch = 0;	
	
	for (int i=0; i<nWeights; i++)
		prevDw(i) = 0;
	
	for (int i=0; i<nWavelons; i++)
		c[i] = rng.uniform(-1.0, 1.0);
	for (int i=0; i<nInputs; i++)
		a[i] = rng.uniform(-0.01, 0.01);
	
	a0 = rng.uniform(-0.01, 0.01);
	
	batch.resize(batchSize);
	batchOut.resize(batchSize);
	batchExact.resize(batchSize);
	
	mu = 1;
	muFactor = 5;
	muMin = 1e-1;
	muMax = 1e+10;
}

void WaveletNet::predict(const vector<double>& inputs, vector<double>& outputs)
{
	double res = 0;
	
	// Bias
	
	//res += a0;
	
	// Linear terms
	
	//for (int i=0; i<nInputs; i++)
	//	res += inputs[i] * a[i];
	
	// Wavelet terms
	
	for (int i=0; i<nWavelons; i++)
		res += wavelons[i]->exec(inputs) * c[i];

	outputs[0] = res;	
}

void WaveletNet::computeJ()
{
	int iw = 0;
	
	// Bias
	
	//J(nInBatch, iw++) = 1;
	
	// Linear terms
	
	//for (int i=0; i < nInputs; i++)
	//	J(nInBatch, iw++) = batch[nInBatch][i];
	
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
	dw = -eta * J.t() * e + alpha * prevDw;
	//cout << J << endl << e << endl << prevDw << endl << dw << endl;
	prevDw = -eta * J.t() * e;
}

void WaveletNetLM::prepareLM()
{
	JtJ = J.t() * J;
	Je  = - J.t() * e;
}

void WaveletNetLM::computeDwLM()
{
	tmp = JtJ + mu*I;
	
	//vec eig;
	//eig_sym(eig, tmp);
	
	//cout << eig(eig.n_elem - 1) / eig(0) << endl;
	//cout << tmp;
	
	//cout << tmp << endl << dw << endl;
	dw = solve(tmp, Je);
	
	//cout << dw << endl;
	dw *= eta;
	
	//cout << J << endl << e << endl << prevDw << endl << dw << endl;
}

void WaveletNet::changeWeights()
{
	// Change the weights
	
	int iw = 0;
	
	// Bias
	
	//a0 += dw(iw++);
	
	// Linear terms
	
	//for (int i=0; i < nInputs; i++)
	//	a[i] += dw(iw++);
	
	// Summing weights
	
	for (int i=0; i < nWavelons; i++)
	{
		w(iw) = c[i] + dw(iw);
		c[i] += dw(iw++);
	}
	
	// Translations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
		{
			w(iw) = wavelons[j]->m[k] + dw(iw);
			wavelons[j]->m[k] += dw(iw++);
		}
	
	// Dilations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
		{
			w(iw) = wavelons[j]->d[k] + dw(iw);
			wavelons[j]->d[k] += dw(iw++);
		}
}

void WaveletNet::rollback()
{
	// Unchange the weights
	
	int iw = 0;
	
	// Bias
	
	//a0 -= dw(iw++);
	
	// Linear terms
	
	//for (int i=0; i < nInputs; i++)
	//	a[i] -= dw(iw++);
	
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
	predict(inputs, tmpVec);
	//batch[nInBatch] = inputs;
	//batchOut[nInBatch] = tmpVec[0];
	//batchExact[nInBatch] = tmpVec[0] - errors[0];
	e(nInBatch) = errors[0];
	
	// Compute grad for this object
	
	computeJ();
	
	nInBatch++;
	
	// If batch is full we have to update
	
	if (nInBatch == batchSize)
	{
		computeDw();
		changeWeights();
		
		nInBatch = 0;
	}		
}

void WaveletNetLM::improve(const vector<double>& inputs, const vector<double>& errors)
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
		prepareLM();
				
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
				dw = -dw;
				changeWeights();
				if (mu < muMax)
				{
					mu *= muFactor;
				}
				else
				{
					break;
				}
			}
		}
		
		if (mu > muMin) mu /= muFactor;
		
		//cout << mu << endl;
		//cout << Q0 << " --> " << Q << endl;
		
		nInBatch = 0;
	}		
}

void WaveletNet::save(string fname)
{
	info("Saving into %s\n", fname.c_str());
	
	string nameBackup = fname + "_tmp";
	ofstream out(nameBackup.c_str());
	
	if (!out.good()) die("Unable to open save into file %s\n", fname.c_str());
	
	out.precision(20);
	
	out << nWeights << " " << nInputs << " " << nWavelons << endl;
	
	//*****************************************
	for (int i=0; i < nWavelons; i++)
		out << c[i] << " ";
	out << endl;
	
	// Translations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			out << wavelons[j]->m[k] << " ";
	out << endl;
	
	// Dilations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			out << wavelons[j]->d[k] << " ";
	out << endl;
	//*****************************************
	
	out.flush();
	out.close();
	
	// Prepare copying command
	string command = "cp ";
	string nameOriginal = fname;
	command = command + nameBackup + " " + nameOriginal;
	
	// Submit the command to the system
	system(command.c_str());
}

bool WaveletNet::restart(string fname)
{
	string nameBackup = fname;
	
	ifstream in(nameBackup.c_str());
	info("Reading from %s\n", nameBackup.c_str());
	if (!in.good())
	{
		error("WTF couldnt open file %s (ok keep going mofo)!\n", fname.c_str());
		return false;
	}
	
	int readNWeights, readNInputs, readNWavelons;
	in >> readNWeights >> readNInputs >> readNWavelons;
	
	if (readNWeights != nWeights || readNInputs != nInputs || readNWavelons != nWavelons)
		die("Network parameters differ!");
	
	//*****************************************
	for (int i=0; i < nWavelons; i++)
		in >> c[i];
	
	// Translations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			in >> wavelons[j]->m[k];
	
	// Dilations
	
	for (int j=0; j < nWavelons; j++)
		for (int k=0; k < nInputs; k++)
			in >> wavelons[j]->d[k];
	//*****************************************
	
	in.close();
	return true;
}


















