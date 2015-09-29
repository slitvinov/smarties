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
    vector<double> prediction;
    vector<double> scaledInp;
    
    RNG rng;
    
    public:
    // Costructor-Destructor
    ANNApproximator(StateInfo newSInfo, ActionInfo newActInfo, string tp, int nAgents);
    ~ANNApproximator();
    
    // Methods
    double get (const State& s, const Action& a, int nAgent = 0);
    double test(const State& s, const Action& a, int nAgent = 0);
    double advance(const State& s, const Action& a, int nAgent = 0);
    
    double getMax (const State& s, int nAgent);
    double testMax (const State& s,int & nAct,  int nAgent);
    double advanceMax (const State& s, int nAgent);
    
    void set (const State& s, const Action& a, double value, int nAgent = 0);
    void correct(const State& s, const Action& a, double error, int nAgent = 0);
    
    double Train() {;}
    void save(string name);
    bool restart(string name);
};
