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

struct Tuple
{
    int agentId;
    State* sOld;
    Action* a;
    State* sNew;
    Real reward;
};

struct NFQdata
{
    vector<Real> insi;
    vector<Real> outi;
    vector<Real> pred;
};

struct Transitions
{
    vector<Tuple> Set;
    StateInfo sInfo;
    ActionInfo actInfo;
    
    Transitions(ActionInfo actInfo, StateInfo sInfo): actInfo(actInfo), sInfo(sInfo) {}
    
    void add(int agentId, State& sOld, Action& a, State& sNew, Real reward)
    {
        Tuple tmp;
        
        tmp.sOld   = new State(sInfo);
        tmp.a      = new Action(actInfo);
        tmp.sNew   = new State(sInfo);
        
        tmp.agentId  = agentId;
        *tmp.sOld  = sOld;
        *tmp.a     = a;
        *tmp.sNew  = sNew;
        tmp.reward = reward;
        
        Set.push_back(tmp); //Growing batch
    }
};


class NFQApproximator : public QApproximator
{
    private:
    int nActions;
    int nStateDims;
    int batchSize;
    StateType sType;
    ActionIterator actionsIt;
    Transitions samples;
    Real gamma;
    vector<Approximator*> ann;
    vector<Real> prediction, A, B;
    vector<Real> scaledInp;
    RNG* rng;
    
    public:
    // Costructor-Destructor
    NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, Real gamma);
    ~NFQApproximator();
    
    // Methods
    Real get (const State& s, const Action& a);
    void set (const State& s, const Action& a, Real value) {;} //nothing to see here
    void correct(const State& s, const Action& a, Real error) {;}
    Real batchUpdate();
    void save(string name);
    bool restart(string name);
    void passData(int agentId, State& sOld, Action& a, State& sNew, Real reward);
    Real descale(Real y, int j) {return (y-B[j])/A[j];}
    Real rescale(Real x, int j) {return  A[j]*x +B[j];}
};
