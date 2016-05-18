/*
 *  TwoFish_Environment.h
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include <vector>
#include "../Util/util.h"
#include "ExternalEnvironment.h"
#include "CellList.h"

//class ExternalAgent;
//#include "../Agents/ExternalAgent.h"

class HardCartEnvironment: public ExternalEnvironment
{
public:
    HardCartEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank);
    void setDims() override;
};


