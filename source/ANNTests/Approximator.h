/*
 *  Approximator.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 04.09.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>

using namespace std;

struct Memory
{
    Memory(int nMems): nMems(nMems)
    {
        memory.resize(nMems);
        cstate.resize(nMems);
    }
    Memory(const Memory& c) : nMems(c.nMems)
    {
        memory.resize(nMems);
        cstate.resize(nMems);
        for (int i = 0; i<nMems; ++i)
        {
            memory[i] = c.memory[i];
            cstate[i] = c.cstate[i];
        }
    }
    Memory operator=(const Memory& c)
    {
        return Memory(c);
    }
    int nMems;
    vector<double> memory;
    vector<double> cstate;
};

class Approximator
{
public:
    vector<Memory> Agents;
	virtual void predict(const vector<double>& inputs, vector<double>& outputs, int nAgent) = 0;
	virtual void improve(const vector<double>& inputs, const vector<double>& errors, int nAgent) = 0;
	virtual void predict(const vector<double>& inputs, const vector<double>& memory_in, const vector<double>& cstate_in,  vector<double>& outputs) = 0;
	virtual void   save(string name)    = 0;
	virtual bool   restart(string name) = 0;
    virtual void   setBatchsize(int size) = 0;
};