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
    double lambdaold, lambdanew, errold, errnew, delta, ALfac; // if >= 1. then i'm not doing advantage learning
    bool first;
    Memory backup;
    ActionIterator actionsIt;
    double gamma, A, B;
    Approximator * ann;
    vector<double> prediction;
    vector<double> scaledInp;
    RNG* rng;
    
    public:
    string nettype;
    // Costructor-Destructor
    NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings settings, int nAgents);
    ~NFQApproximator();
    
    // Methods
    double get (const State& s, const Action& a, int nAgent = 0);
    double test(const State& s, const Action& a, int nAgent = 0);
    double advance(const State& s, const Action& a, int nAgent = 0);
    double getMax (const State& s, int & nAct, int nAgent);
    double testMax (const State& s, int & nAct,  int nAgent);
    double advanceMax (const State& s, int & nAct, int nAgent);
    
    void set (const State& s, const Action& a, double value, int nAgent = 0) {;} //nothing to see here
    
    void correct(const State& s, const Action& a, double error, int nAgent = 0);
    
    double Train()
    {
        if (nettype == "LSTM")
            return serialUpdate();
        else
            return batchUpdate();
    }
    double batchUpdate();
    double serialUpdate();
    //double serialALearning();
    void save(string name);
    bool restart(string name);
    void passData(int agentId, State& sOld, Action& a, State& sNew, double reward, double altrew);
    double descale(double y) {return (y-B)/A;}//y;}//
    double rescale(double x) {return  A*x +B;}//x;}//
};
