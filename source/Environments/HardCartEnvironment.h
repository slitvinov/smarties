/*
 *  HardCartEnvironment.h
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include <vector>

#include "Environment.h"
#include "CellList.h"

class HardCartAgent;
#include "../Agents/HardCartAgent.h"

class HardCartEnvironment: public Environment
{
    string execpath;
    virtual void setDims();
    vector<int> ids;
    
    FILE *fin, *fout;
    
    vector<HardCartAgent*> exagents;
    vector<State>  states;
    vector<Real> rewards;
    vector<Action> actions;

public:
    HardCartEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, int index);
    int   evolve   (Real t);
};


