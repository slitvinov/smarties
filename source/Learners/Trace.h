/*
 *  Trace.h
 *  smarties
 *
 *  Created by Dmitry Alexeev on Jun 9, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include <vector>

#include "../StateAction.h"

#pragma once

using namespace std;

struct History
{
    State* s;
    Action* a;
    double value;
};

struct Trace
{
    vector<History> hist;
    int len, start;

    void add(State& s, Action& a)
    {
        start = (start == len-1) ? 0 : start + 1;
        *hist[start].s = s;
        *hist[start].a = a;
        hist[start].value = 1;
    }
};



