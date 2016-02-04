/*
 * ANNApproximator.h
 * rl
 *
 * Created by Dmitry Alexeev on 24.06.13.
 * Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <map>
#include <string>
#include <vector>

#include "../ANN/Approximator.h"
#include "QApproximator.h"
#include "../rng.h"

class ANNApproximator : public QApproximator
{
    private:
    int nActions, nInputs;
    int nStateDims;
    int batchSize;
    int nAgents;
    
    string nettype;
    Memory backup;
    Approximator * ann;
    vector<Real> prediction;
    vector<Real> scaledInp;
    
    RNG rng;
    
    public:
    // Costructor-Destructor
    ANNApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings settings, int nAgents);
    ~ANNApproximator();
    
    // Methods
    Real get (const State& s, const Action& a, int nAgent = 0);
    Real test(const State& s, const Action& a, int nAgent = 0);
    Real advance(const State& s, const Action& a, int nAgent = 0);
    
    Real getMax (const State& s, int & nAct, int nAgent);
    Real testMax (const State& s, int & nAct,  int nAgent);
    Real advanceMax (const State& s, int & nAct, int nAgent);
    
    void set (const State& s, const Action& a, Real value, int nAgent = 0);
    void correct(const State& s, const Action& a, Real error, int nAgent = 0);
    
    Real Train() {;}
    void save(string name);
    bool restart(string name);
};
