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
//#include "../QApproximators/ANNApproximator.h"

using namespace std;

class QLearning : public Learner
{
private:
	QApproximator* Q;
	ActionIterator actionsIt;
	map<string, QApproximator*> QMap;
		
	double gamma, greedyEps, lRate;
	
	RNG* rng;
    
    int suffix;
		
public:
	QLearning(QApproximator* newQ, ActionInfo& actInfo, double newGamma, double newGreedyEps, double newLRate);
	
    void selectAction(State& s, Action& a);
    void update(State& sOld, Action& a, double r, State& s);
    
    void try2restart(string prefix);
    void savePolicy(string fname);
};

