/*
 *  CartEnvironment.h
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

class CartEnvironment: public Environment
{
    string execpath;
    void setDims();
    vector<int> ids;

    FILE *fin, *fout;

    vector<CartAgent*> exagents;
    vector<State>  states;
    vector<Real> rewards;
    vector<Action> actions;

public:
   CartEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, int index);

   int   evolve   (Real t);
};


