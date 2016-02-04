/*
 *  Explorer.h
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

class Explorer : public Learner
{
private:
	QApproximator* Q;
    QApproximator* errEst;
	ActionIterator actionsIt;
	map<string, QApproximator*> QMap;
		
	double gamma, greedyEps, lRate;
	
	RNG* rng;
    
    int suffix;
		
public:
	Explorer(QApproximator* newQ, QApproximator* errQ, ActionInfo& actInfo, double newGamma, double newGreedyEps, double newLRate, double lambda);
	
    void updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, double r, int Nagent = 0);
    void updateSelect(Trace& t, State& s, Action& a, double r, double alphaK) {};
    void try2restart(string prefix);
    void savePolicy(string fname);
};

