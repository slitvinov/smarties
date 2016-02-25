//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#pragma once
#define MEGADEBUG
#include "../Learners/Learner.h"
#include "../Savers/Saver.h"
#include "../Learners/Trace.h"
#include "../QApproximators/MultiTable.h"
#include "../QApproximators/QApproximator.h"
#include "../QApproximators/NFQApproximator.h"

#ifndef MEGADEBUG
#include <mpi.h>
#endif

class Saver;

class Master
{
private:
    Learner* learner;
    QApproximator* Q;
    int insize,  inOneSize, nInfo;
    int outsize, outOneSize;
    byte *inbuf, *outbuf;
    
    inline void unpackChunk(byte* &buf, int & first, State& sOld, Action& a, Real& r, vector<Real>& info, State& s);
    inline void packChunk(byte* &buf, Action a);
    
    Real totR;
    vector<Saver*> savers;
    
    int nAgents, nSlaves;
    vector<Trace> traces;

    void execSavers(Real time, int iter);

public:
    StateInfo  sInfo;
    ActionInfo actInfo;
    Master(Learner* learner, QApproximator* newQ, Environment* env, int nSlaves, Real traceDecay);
    Real getTotR() { Real tmp = totR; totR = 0; return tmp; }
    void restart(string fname);
    
    void run();

    void registerSaver(Saver* saver);
};

class Slave
{
    int me;
    
    Environment* env;
    vector<Agent*> agents;
    Real dt;
    int first;
    int insize, outsize, nInfo;
    byte *inbuf, *outbuf;
    
    vector<Action> actions;
    vector<State> oldStates;
    bool* needToPack;
    void packData();
    void unpackData();
        
public:
    
    Slave(Environment* env, Real newDt, int me);
    void evolve(Real& t);
    
};
