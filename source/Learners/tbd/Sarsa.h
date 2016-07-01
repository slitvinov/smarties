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
    vector<Trace> traces;
	Real lambda;
		
public:
    Sarsa(Environment* env, Settings & settings);	
    void updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r) override;
};

