/*
 *  Learner.h
 *  rl
 *
 *  Created by Dmitry Alexeev on 15.07.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include <vector>
#include <string>
#include <list>

#include "../Agents/Agent.h"
#include "../Environments/Environment.h"

#include "../QApproximators/MultiTable.h"
#include "../QApproximators/QApproximator.h"
#include "../QApproximators/NFQApproximator.h"

#include "../Transitions/Transitions.h"
#include "Trace.h"

using namespace std;

class Learner
{
protected:
    int suffix, nAgents;
    Real gamma, greedyEps;
    ActionInfo aInfo;
    StateInfo  sInfo;
    
    RNG* rng;
    QApproximator* Q;
public:
    Transitions* T;
    
    Learner(Environment* env, Settings & settings) :
    suffix(0), nAgents(settings.nAgents), gamma(settings.gamma), greedyEps(settings.greedyEps), aInfo(env->aI), sInfo(env->sI)
    {
        rng = new RNG(rand());
        
        if (settings.approx == "NN")
            Q = new NFQApproximator(sInfo, aInfo, settings);
        else if (settings.approx == "table")
            Q = new MultiTable(sInfo, aInfo, settings);
        else {die("Undefined approximator.\n");}
        
        T = new Transitions(env, settings);
    };

    virtual void updateSelect(const int agentId, State& s, Action& a, State& sOld, Action& aOld, vector<Real> info, Real r) = 0;
    
    virtual void Train() {};
    
    void try2restart(string fname)
    {
        _info("Restarting from saved policy...\n");
        
        T->restartSamples();
        if ( Q->restart(fname) ) {_info("Restart successful, moving on...\n");}
        else {_info("Not all policies restarted, therefore assumed zero. Moving on...\n");}
    }
    
    void savePolicy(string fname)
    {
        _info("\nSaving all policies...\n");
        Q->save(fname);
        _info("Done\n");
    }
};
