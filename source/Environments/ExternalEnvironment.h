/*
 *  ExternalEnvironment.h
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

class ExternalEnvironment: public Environment
{
    string execpath;
    void setDims();
    vector<int> ids;
    int pid;
    FILE *fin, *fout;

    vector<ExternalAgent*> exagents;
    vector<State>  states;
    vector<Real> rewards;
    vector<Action> actions;

public:
   ExternalEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, int index);

   int   evolve   (Real t);
};


