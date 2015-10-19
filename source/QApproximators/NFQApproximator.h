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

<<<<<<< HEAD
=======
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
        
        tmp.agentId= agentId;
        *tmp.sOld  = sOld;
        *tmp.a     = a;
        *tmp.sNew  = sNew;
        tmp.reward = reward;
        
        Set.push_back(tmp); //Growing batch
    }
};


>>>>>>> 10c86dd8b1b6ba41f121230dc7ac9472ac58440d
class NFQApproximator : public QApproximator
{
    private:
    int nInputs;
    int batchSize;
    int nAgents;
    double lambdaold, lambdanew, errold, errnew, delta;
    bool first;
<<<<<<< HEAD
    Memory backup;
    ActionIterator actionsIt;
=======
    
    ActionIterator actionsIt;
    Transitions samples;
>>>>>>> 10c86dd8b1b6ba41f121230dc7ac9472ac58440d
    double gamma, A, B;
    Approximator * ann;
    vector<double> prediction;
    vector<double> scaledInp;
    RNG* rng;
    
    public:
    string nettype;
    // Costructor-Destructor
    NFQApproximator(StateInfo newSInfo, ActionInfo newActInfo, double gamma, string nettype, int nAgents);
    ~NFQApproximator();
    
    // Methods
    double get (const State& s, const Action& a, int nAgent = 0);
    double test(const State& s, const Action& a, int nAgent = 0);
<<<<<<< HEAD
    double advance(const State& s, const Action& a, int nAgent = 0);
    double getMax (const State& s, int nAgent);
    double testMax (const State& s,int & nAct,  int nAgent);
    double advanceMax (const State& s, int nAgent);
    
    void set (const State& s, const Action& a, double value, int nAgent = 0) {;} //nothing to see here
    
    void correct(const State& s, const Action& a, double error, int nAgent = 0);
=======
   
    double advance(const State& s, const Action& a, int nAgent = 0)
    {
        return get(s, a, nAgent);
    }
    void set (const State& s, const Action& a, double value, int nAgent = 0) {;} //nothing to see here
    
    void correct(const State& s, const Action& a, double error, int nAgent = 0) {;}
>>>>>>> 10c86dd8b1b6ba41f121230dc7ac9472ac58440d
    
    double Train()
    {
        if (nettype == "LSTM")
<<<<<<< HEAD
        {
            if(prediction.size() < 2)
                return serialUpdate();
            return serialALearning();
        }
=======
            return serialUpdate();
>>>>>>> 10c86dd8b1b6ba41f121230dc7ac9472ac58440d
        else
            return batchUpdate();
    }
    double batchUpdate();
    double serialUpdate();
<<<<<<< HEAD
    double serialALearning();
=======
>>>>>>> 10c86dd8b1b6ba41f121230dc7ac9472ac58440d
    void save(string name);
    bool restart(string name);
    void passData(int agentId, State& sOld, Action& a, State& sNew, double reward, double altrew);
    double descale(double y) {return (y-B)/A;}
    double rescale(double x) {return  A*x +B;}
};
