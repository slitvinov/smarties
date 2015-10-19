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
    Memory(int nMems=1, int nRecurr=1): nMems(nMems), nRecurr(nRecurr)
    {
        memory.resize(nRecurr);
        ostate.resize(nMems);
        nstate.resize(nMems);
    }
    Memory(const Memory& c) : nMems(c.nMems), nRecurr(c.nRecurr)
    {
        ostate.resize(nMems);
        nstate.resize(nMems);
        memory.resize(nRecurr);
        for (int i = 0; i<c.nMems; ++i)
        {
            ostate[i] = c.ostate[i];
            nstate[i] = c.nstate[i];
        }
        for (int i = 0; i<c.nRecurr; ++i)
            memory[i] = c.memory[i];
    }
    void copy(const Memory& c)
    {
        nMems = c.nMems;
        nRecurr = c.nRecurr;
        ostate.resize(c.nMems);
        nstate.resize(c.nMems);
        memory.resize(c.nRecurr);
        for (int i = 0; i<c.nMems; ++i)
        {
            ostate[i] = c.ostate[i];
            nstate[i] = c.nstate[i];
        }
        for (int i = 0; i<c.nRecurr; ++i)
            memory[i] = c.memory[i];
    }
    void init(int _nMems, int _nRecurr)
    {
        nMems = _nMems;
        nRecurr = _nRecurr;
        ostate.resize(nMems);
        nstate.resize(nMems);
        memory.resize(nRecurr);
    }
    int nMems, nRecurr;
    vector<double> memory;
    vector<double> nstate;
    vector<double> ostate;
};

class Approximator
{
public:
    double lambda;
    vector<Memory> Agents;
	virtual void predict(const vector<double>& input, vector<double>& output, int nAgent) = 0;
	virtual void improve(const vector<double>& input, const vector<double>& error, int nAgent) = 0;
	virtual void predict(const vector<double>& input, const vector<double>& memoryin, const vector<double>& ostate, vector<double>& nstate,  vector<double>& output) = 0;
	virtual void   save(string name)    = 0;
	virtual bool   restart(string name) = 0;
    virtual void   setBatchsize(int size) = 0;
    virtual double TotSumWeights() {return 0;}
    virtual double AvgLearnRate() {return 0;}
};