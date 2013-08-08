/*
 *  ANNApproximator.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 24.06.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <map>
#include <string>
#include <vector>

#include "../ANN/Network.h"
#include "../ANN/WaveletNet.h"
#include "QApproximator.h"
#include "../rng.h"

class ANNApproximator : public QApproximator
{
public:
	int nActions;
	int nStateDims;
	int batchSize;
	vector<WaveletNet*> ann;
	vector<double> prediction;
	vector<double> scaledInp;
	
	vector< vector< vector<double> > > batch;   // Num of actions * batch size * state dimension
	
	RNG rng;
	
public:
	// Costructor-Destructor
	ANNApproximator(StateInfo newSInfo, ActionInfo newActInfo);
	~ANNApproximator();
	
	// Methods
	double get(const State& s, const Action& a);
	void   set(const State& s, const Action& a, double value);
	
	void   save(string name)    { };
	bool   restart(string name) { return false; };
};