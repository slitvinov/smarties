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
    double reward;
};

struct NFQdata
{
    vector<double> insi;
    vector<double> outi;
    vector<double> pred;
};

struct Transitions
{
    vector<Tuple> Set;
    StateInfo sInfo;
    ActionInfo actInfo;
    
    Transitions(ActionInfo actInfo, StateInfo sInfo): actInfo(actInfo), sInfo(sInfo) {}
    
    void add(int agentId, State& sOld, Action& a, State& sNew, double reward)
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
    double gamma;
    vector<Approximator*> ann;
    vector<double> prediction, A, B;
    vector<double> scaledInp;
    RNG* rng;
    
    public:
    // Costructor-Destructor
    NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, double gamma);
    ~NFQApproximator();
    
    // Methods
    double get (const State& s, const Action& a);
    void set (const State& s, const Action& a, double value) {;} //nothing to see here
    void correct(const State& s, const Action& a, double error) {;}
    double batchUpdate();
    void save(string name);
    bool restart(string name);
    void passData(int agentId, State& sOld, Action& a, State& sNew, double reward);
    double descale(double y, int j) {return (y-B[j])/A[j];}
    double rescale(double x, int j) {return  A[j]*x +B[j];}
};
