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

#include "Learner.h"
#include "../QApproximators/MultiTable.h"
#include "../QApproximators/QApproximator.h"
#include "Trace.h"

using namespace std;

class Sarsa : public Learner
{
private:
	QApproximator* Q;
	ActionIterator actionsIt;
	map<string, QApproximator*> QMap;
		
	double gamma, greedyEps, lRate, lambda;
	
	RNG* rng;
    
    int suffix;
		
public:
    Sarsa(QApproximator* Q, ActionInfo& actInfo, double gamma, double greedyEps, double lRate, double lambda);
	
    void updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, double r, int Nagent = 0);
    void updateSelect(Trace& t, State& s, Action& a, double r, double alphaK) {};
    void try2restart(string prefix);
    void savePolicy(string fname);
};

