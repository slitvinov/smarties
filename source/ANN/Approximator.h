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
#include "../Types/types.h"
#define SIMD 1 //TODO
using namespace std;
typedef float vt;

struct Memory
{
    Memory(int _nNeurons=1, int _nStates=1): nNeurons(_nNeurons), nStates(_nStates)
    {
        oldvals = (vt*) _mm_malloc(nNeurons*sizeof(vt), ALLOC);
        ostates = (vt*) _mm_malloc(nStates *sizeof(vt), ALLOC);
    }
    
    ~VelocitySolverNSquared()
    {
        _mm_free(outvals);
        _mm_free(ostates);
    }
    
    void init(int _nNeurons, int _nStates)
    {
        _mm_free(oldvals);
        _mm_free(ostates);
        nNeurons = _nNeurons;
        nStates = _nStates;
        oldvals = (vt*) _mm_malloc(nNeurons*sizeof(vt), ALLOC);
        ostates = (vt*) _mm_malloc(nStates *sizeof(vt), ALLOC);
    }
    
    int nNeurons, nStates;
    vt * oldvals;
    vt * ostates;
};


class Approximator
{
public:
    vector<Memory> Agents;
    
    virtual void predict(const vector<vt>& input, vector<vt>& output, int nAgent) {};
	virtual void improve(const vector<vt>& input, const vector<vt>& error, int nAgent) {};
    virtual void test(const vector<vt>& input, vector<vt>& output, int nAgent) { predict(input, output, nAgent); };
	virtual void save(string name) = 0;
	virtual bool restart(string name) = 0;
    virtual void setBatchsize(int size) = 0;
    virtual vt TotSumWeights() {return 0;}
    virtual vt AvgLearnRate() {return 0;}
};
