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

#include "ExternalEnvironment.h"
#include "CellList.h"

class ExternalAgent;
#include "../Agents/ExternalAgent.h"

class HuntEnvironment: public ExternalEnvironment
{
    void setDims();
    
public:
   HuntEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, int index) : ExternalEnvironment(agents, execpath, tp, rank, index) { }
};


