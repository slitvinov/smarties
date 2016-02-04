/*
 * NFQApproximator.h
 * rl
 *
 * Created by Guido Novati on 16.07.15.
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


class NFQApproximator : public QApproximator
{
    private:
    int nInputs;
    int batchSize;
    int nAgents;
    Real lambdaold, lambdanew, errold, errnew, delta, ALfac; // if >= 1. then i'm not doing advantage learning
    bool first;
    Memory backup;
    ActionIterator actionsIt;
    Real gamma, A, B;
    Approximator * ann;
    vector<Real> prediction;
    vector<Real> scaledInp;
    RNG* rng;
    
    public:
    string nettype;
    // Costructor-Destructor
    NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings settings, int nAgents);
    ~NFQApproximator();
    
    // Methods
    Real get (const State& s, const Action& a, int nAgent = 0);
    Real test(const State& s, const Action& a, int nAgent = 0);
    Real advance(const State& s, const Action& a, int nAgent = 0);
    Real getMax (const State& s, int & nAct, int nAgent);
    Real testMax (const State& s, int & nAct,  int nAgent);
    Real advanceMax (const State& s, int & nAct, int nAgent);
    
    void set (const State& s, const Action& a, Real value, int nAgent = 0) {;} //nothing to see here
    
    void correct(const State& s, const Action& a, Real error, int nAgent = 0);
    
    Real Train()
    {
        if (nettype == "LSTM")
            return serialUpdate();
        else
            return batchUpdate();
    }
    Real batchUpdate();
    Real serialUpdate();
    //Real serialALearning();
    void save(string name);
    bool restart(string name);
    void passData(int agentId, State& sOld, Action& a, State& sNew, Real reward, Real altrew);
    Real descale(Real y) {return (y-B)/A;}//y;}//
    Real rescale(Real x) {return  A*x +B;}//x;}//
};
