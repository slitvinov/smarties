/*
 *  NFQ.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
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
#include "../QApproximators/NFQApproximator.h"
#include "Trace.h"
#include "../QApproximators/ANNApproximator.h"

using namespace std;

class NFQ : public Learner
{
private:
	QApproximator* Q;
	ActionIterator actionsIt;
	map<string, QApproximator*> QMap;
		
	double gamma, greedyEps, lRate;
	
	RNG* rng;
    
    int suffix;
		
public:
	NFQ(QApproximator* newQ, ActionInfo& actInfo, double newGamma, double newGreedyEps, double newLRate);
	
    void updateSelect(Trace& t, State& s, Action& a, State& sOld, Action& aOld, double r, int Nagent = 0);
    void updateSelect(Trace& t, State& s, Action& a, double r, double alphaK) {};
    void try2restart(string prefix);
    void savePolicy(string fname);
    void NFQimprove();
};

