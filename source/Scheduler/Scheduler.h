//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#pragma once
//#define MEGADEBUG
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
    int insize,  inOneSize, nInfo;
    int outsize, outOneSize;
    byte *inbuf, *outbuf;
    bool bTRAINING;
    inline void unpackChunk(byte* &buf, int & first, State& sOld, Action& a, Real& r, vector<Real>& info, State& s);
    inline void packChunk(byte* &buf, Action a);
    
    Real totR;
    vector<Saver*> savers;
    
    int nAgents, nSlaves;

    void execSavers(Real time, int iter);

public:
    StateInfo  sInfo;
    ActionInfo actInfo;
    Master(Learner* learner, Environment* env, Settings & settings);
    Real getTotR() { Real tmp = totR; totR = 0; return tmp; }
    void restart(string fname);
    void run();
    void registerSaver(Saver* saver);
};

class Slave
{
    int me;
    RNG* rng;
    Environment* env;
    vector<Agent*> agents;
    bool bTRAINING;
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
    
    Slave(Environment* env, int me, Settings & settings);
    void evolve(Real& t);
    void restart(string fname); //TODO
    Learner* learner; //TODO
    
};
