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

#include "ANN/Network.h"
#include "QApproximator.h"

class ANNApproximator : public QApproximator
{
public:
	int nActions;
	int nStateDims;
	vector<Network*> ann;
	vector<double> prediction;
	vector<double> scaledInp;
	
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