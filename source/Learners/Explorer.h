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
#include "../QApproximators/NFQApproximator.h"

using namespace std;

class Explorer : public Learner
{
public:
	Explorer(Environment* env, Settings & settings);
    vector<Real> max_scale, min_scale;
    void updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r) override;
};

