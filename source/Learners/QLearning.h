/*
 *  QLearning.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <string>
#include <list>

#include "Learner.h"
#include "../QApproximators/MultiTable.h"
#include "../QApproximators/QApproximator.h"
#include "../QApproximators/ANNApproximator.h"

using namespace std;

class QLearning : public Learner
{
private:
	vector<QApproximator*> Qarray;
	vector<State> s0, s1;
	vector<Action> a0, a1;
	vector<double> r;
	vector<double> bestActionVals;
	vector<ActionIterator> actionsIt;
	map<string, QApproximator*> QMap;
		
	double gamma, greedyEps, lRate;
	
	RNG* rng;
	
	void agentsChoose(double t);
	void agentsUpdate(double t);
	void agentsAct(double t);
	void agentsMove();
		
public:
	QLearning(System newSystem, double newGamma, double newGreedyEps, double newLRate, double newDt, MRAG::Profiler* newProfiler = NULL);
	void evolve(double t);
	void savePolicy(string prefix);
	void try2restart(string prefix);
};

