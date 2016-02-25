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
    int nInputs, nActions, nStateDims;
    int batchSize, iter;
    Real delta, ALfac; // if >= 1. then i'm not doing advantage learning
    bool first;
    ActionIterator actionsIt;
    Real gamma, A, B;
    Approximator * ann;
    vector<Real> prediction;
    vector<Real> scaledInp;
    vector<int> indexes;
    RNG* rng;
    
    public:
    string nettype;
    // Costructor-Destructor
    NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, Settings & settings, int nAgents);
    ~NFQApproximator();
    
    // Methods
    void get(const State& sOld, vector<Real> & Qold, const State& s, vector<Real> & Q, int iAgent = 0) override;
    Real get (const State& s, const Action& a, int nAgent = 0) override;
    Real getMax (const State& s, Action& a, int nAgent = 0) override;
    
    void set (const State& s, const Action& a, Real value, int nAgent = 0) override
    {
        correct(s, a, value, nAgent);
    } //nothing to see here
    
    void correct(const State& s, const Action& a, Real error, int nAgent = 0) override;
    
    void Train() override;
    //Real batchUpdate();
    //Real serialUpdate();
    
    void passData(int & agentId, int & first, State & sOld, Action & a, State & sNew, Real & reward, vector<Real>& info) override
    {
        QApproximator::passData(agentId, first, sOld, a, sNew, reward, info);
        if (first)
            ann->resetMemories(agentId);
    }
    
    void save(string name) override;
    bool restart(string name) override;
    Real descale(Real y) {return (y-B)/A;}//y;}//
    Real rescale(Real x) {return  A*x +B;}//x;}//
};
