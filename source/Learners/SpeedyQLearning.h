/*
 *  SpeedyQLearning.h
 *  rl
 *	enhances convergence rate of classic Q learning
 *  Created by Gabriele Abbati on 14/07/2015.
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <string>
#include <list>

#include "Learner.h"
#include "../QApproximators/MultiTable.h"
#include "../QApproximators/QApproximator.h"
#include "Trace.h"
//#include "../QApproximators/ANNApproximator.h"

using namespace std;

class SpeedyQLearning : public Learner
{
private:
	// approximator for timestep k Q-values
	QApproximator* Q;
	// approximator for timestep k-1 Q-values
	QApproximator* Qold;
	ActionIterator actionsIt;
	map<string, QApproximator*> QMap;
		
	double gamma, greedyEps, lRate, lambda;
	
	RNG* rng;
    
    int suffix;
		
public:
	SpeedyQLearning(QApproximator* newQ, QApproximator* oldQ, ActionInfo& actInfo, double newGamma, double newGreedyEps, double newLRate, double LAMBDA);
	
	void updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, double r, int Nagent = 0);
    void updateSelect(Trace& t, State& s, Action& a, double r, double alphaK);
    void try2restart(string prefix);
    void savePolicy(string fname);
};