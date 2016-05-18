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
#include "Trace.h"

using namespace std;

class NFQ : public Learner
{
public:
	NFQ(Environment* env, Settings & settings);
    bool bTRAINING;
    void updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r) override;
    void Train() override;
};

