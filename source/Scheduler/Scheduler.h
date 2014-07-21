//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#ifndef __rl__Scheduler__
#define __rl__Scheduler__

#include "../Learners/Learner.h"
#include "../Environments/Environment.h"

#include <mpi.h>

class Master
{
    ActionInfo actInfo;
    StateInfo  sInfo;
    Learner* learner;
    
    int insize,  inOneSize;
    int outsize, outOneSize;
    byte *inbuf, *outbuf;
    
    inline void unpackChunk(byte* &buf, State& sOld, Action& a, double& r, State& s);
    inline void packChunk(byte* &buf, Action a);
    
    double totR;
    
public:
    Master(Learner* learner, ActionInfo actInfo, StateInfo sInfo);
    double getTotR() { double tmp = totR; totR = 0; return tmp; }
    void restart(string fname);
    
    void run();
};

class Slave
{
    int me;
    
    System system;
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
    
    Slave(System& newSystem, double newDt, int me);
    void evolve(double& t);
};

#endif /* defined(__rl__Scheduler__) */
