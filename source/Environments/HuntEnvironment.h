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

<<<<<<< HEAD
#include "ExternalEnvironment.h"
=======
#include "Environment.h"
>>>>>>> 10c86dd8b1b6ba41f121230dc7ac9472ac58440d
#include "CellList.h"

class ExternalAgent;
#include "../Agents/ExternalAgent.h"

<<<<<<< HEAD
class HuntEnvironment: public ExternalEnvironment
{
    void setDims();
    
public:
   HuntEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, int index) : ExternalEnvironment(agents, execpath, tp, rank, index) { }
=======
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
>>>>>>> 10c86dd8b1b6ba41f121230dc7ac9472ac58440d
};


