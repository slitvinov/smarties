/*
 *  HuntEnvironment.h
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

class ExternalAgent;
#include "../Agents/ExternalAgent.h"

class HuntEnvironment: public Environment
{
    string execpath;
    void setDims();
    vector<int> ids;

    FILE *fin, *fout;

    vector<ExternalAgent*> exagents;
    vector<State>  states;
    vector<double> rewards;
    vector<Action> actions;

public:
   HuntEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, int index);

   int   evolve   (double t);
};


