//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#pragma once

#include "../Learners/Learner.h"
#include "../Savers/Saver.h"
#include "../Learners/Trace.h"

#include <mpi.h>

class Saver;

class Master
{
public:
    ActionInfo actInfo;
    StateInfo  sInfo;
    Learner* learner;
    
    int insize,  inOneSize;
    int outsize, outOneSize;
    byte *inbuf, *outbuf;
    
    inline void unpackChunk(byte* &buf, State& sOld, Action& a, double& r, State& s);
    inline void packChunk(byte* &buf, Action a);
    
    double totR;
    vector<Saver*> savers;
    
    int nAgents, nSlaves;
    vector<Trace> traces;

    void execSavers(double time, int iter);

public:
    Master(Learner* learner, ActionInfo actInfo, StateInfo sInfo, int nAgents, int nSlaves, double traceDecay);
    double getTotR() { double tmp = totR; totR = 0; return tmp; }
    void restart(string fname);
    
    void run();

    void registerSaver(Saver* saver);
};

class Slave
{
    int me;
    
    Environment* env;
    vector<Agent*> agents;
    double dt;
    
    int insize, outsize;
    byte *inbuf, *outbuf;
    
    vector<Action> actions;
    vector<State> oldStates;
    bool* needToPack;
    
    void packData();
    void unpackData();
        
public:
    
    Slave(Environment* env, double newDt, int me);
    void evolve(double& t);
};
