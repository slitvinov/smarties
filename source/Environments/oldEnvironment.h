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

//class Agent;
//#include "../Agents/Agent.h"

class oldEnvironment: public Environment
{
    string execpath;
    void setDims();
    vector<int> ids;
    int pid, n, rank;
    FILE *fin, *fout;

    //vector<Agent*> exagents;
    vector<State>  states;
    vector<Real> rewards;
    vector<Action> actions;

public:
   oldEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank);
    void setup_Comm() override;
    int   evolve   (Real t) override;
    int   init   () override;
};


