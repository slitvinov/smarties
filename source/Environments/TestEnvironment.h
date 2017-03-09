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

class TestEnvironment : public Environment
{
public:
    TestEnvironment(const int nAgents, const string execpath,
                    const int _rank, Settings & settings);

    void setDims() override;
};