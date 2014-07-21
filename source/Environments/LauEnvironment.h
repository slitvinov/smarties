/*
 *  LauEnvironment.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

#include "Environment.h"

class SmartyLauFish;
#include "../Agents/SmartyLauFish.h"
#include "../../MRAGapps/IF2D_ROCKS/source/IF2D_FluidMediatedLau.h"

class LauEnvironment : public ::Environment
{
private:
	void setDims();
	
public:
    IF2D_FluidMediatedLau* fluid;

	LauEnvironment(vector<Agent*> newAgents, const int argc, const char** argv);

	double getReward();
    double getState();
    void   act(int a);
};
