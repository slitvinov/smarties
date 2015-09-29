/*
 *  GlideEnvironment.h
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

class CartAgent;
#include "../Agents/CartAgent.h"

class GlideEnvironment: public Environment
{
    string execpath;
    void setDims();
    vector<int> ids;
    int pid;
    
    FILE *fin, *fout;

    vector<CartAgent*> exagents;
    vector<State>  states;
    vector<double> rewards;
    vector<Action> actions;

public:
   GlideEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, int index);

   int   evolve   (double t);
};


