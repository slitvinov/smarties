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
#include "Trace.h"
#include "../QApproximators/ANNApproximator.h"

using namespace std;

class QLearning : public Learner
{
private:
	QApproximator* Q;
	ActionIterator actionsIt;
	map<string, QApproximator*> QMap;
		
	Real gamma, greedyEps, lRate;
	
	RNG* rng;
    
    int suffix;
		
public:
	QLearning(QApproximator* newQ, ActionInfo& actInfo, Real newGamma, Real newGreedyEps, Real newLRate);
	
    void updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, Real r, int Nagent = 0);
    void updateSelect(Trace& t, State& s, Action& a, Real r, Real alphaK) {};
    void try2restart(string prefix);
    void savePolicy(string fname);
};

