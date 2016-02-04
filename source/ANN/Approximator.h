/*
 *  Approximator.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 04.09.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include <immintrin.h>
#include <vector>
#include "../Settings.h"
#define SIMD 1 //TODO
#define ALLOC 64
using namespace std;

struct Memory
{
    Memory(int _nNeurons=1, int _nStates=1): nNeurons(_nNeurons), nStates(_nStates)
    {
        oldvals = (Real*) _mm_malloc(nNeurons*sizeof(Real), ALLOC);
        ostates = (Real*) _mm_malloc(nStates *sizeof(Real), ALLOC);
    }
    
    ~Memory()
    {
        _mm_free(oldvals);
        _mm_free(ostates);
    }
    
    void init(int _nNeurons, int _nStates)
    {
        _mm_free(oldvals);
        _mm_free(ostates);
        nNeurons = _nNeurons;
        nStates = _nStates;
        oldvals = (Real*) _mm_malloc(nNeurons*sizeof(Real), ALLOC);
        ostates = (Real*) _mm_malloc(nStates *sizeof(Real), ALLOC);
    }
    
    int nNeurons, nStates;
    Real * oldvals;
    Real * ostates;
};


class Approximator
{
public:
    vector<Memory> Agents;
    
    virtual void predict(const vector<Real>& input, vector<Real>& output, int nAgent) {};
	virtual void improve(const vector<Real>& input, const vector<Real>& error, int nAgent) {};
    virtual void test(const vector<Real>& input, vector<Real>& output, int nAgent) { predict(input, output, nAgent); };
	virtual void save(string name) = 0;
	virtual bool restart(string name) = 0;
    virtual void setBatchsize(int size) = 0;
    virtual Real TotSumWeights() {return 0;}
    virtual Real AvgLearnRate() {return 0;}
};
