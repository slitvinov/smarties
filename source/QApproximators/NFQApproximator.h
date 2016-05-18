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

#include "../ANN/Network.h"
#include "../ANN/Optimizer.h"
#include "QApproximator.h"
#include "../Profiler.h"


class NFQApproximator : public QApproximator
{
    private:
    int nInputs, nActions, nStateDims;
    int batchSize, iter;
    bool first;

    vector<Real> prediction, scaledInp;
    vector<int> indexes;
    RNG* rng;
    Network* net;
    Optimizer* opt;
    public:
    NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings & settings);
    ~NFQApproximator();
    
    Profiler* profiler;
    // Methods
    void get(const State& sOld, vector<Real> & Qold, const State& s, vector<Real> & Q, int iAgent = 0) override;
    Real get (const State& s, const Action& a, int nAgent = 0) override;
    Real getMax (const State& s, Action& a, int nAgent = 0) override;
    
    void set (const State& s, const Action& a, Real value, int nAgent = 0) override
    {
        die("nothing to see here\n");//correct(s, a, value, nAgent);
    }
    
    void correct(const State& s, const Action& a, Real error, int nAgent = 0) override;
    Real Train(const vector<vector<Real>> & sOld, const vector<int> & a, const vector<Real> & r, const vector<vector<Real>> & s, Real gamma, Real weight=1.) override;
    void updateFrozenWeights() override
    {
        net->updateFrozenWeights();
        cout << profiler->printStat() << endl;
    }
    void save(string name) override;
    bool restart(string name) override;
    //Real descale(Real y) {return (y-B)/A;}
    //Real rescale(Real x) {return  A*x +B;}
};
