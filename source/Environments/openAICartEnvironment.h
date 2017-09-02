/*
 *  Environment.h
 *  rl
 *
 *  Created by Guido Novati on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "Environment.h"

class openAICartEnvironment : public Environment
{
   const bool allSenses;
public:
    openAICartEnvironment(const int nAgents, const string execpath, Settings & settings);

    //void setDims() override;
};
