//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#pragma once

class Learner;

#include "Learners/Learner.h"


class Master
{
private:
    Learner* const learner;
    Environment* const env;
    ActionInfo actInfo;
    StateInfo  sInfo;
    const bool bTrain;
    int nAgents, nSlaves, saveFreq, iter, inOneSize, outOneSize;
    mt19937 * gen;
    State  sOld, s;
    Action aOld, a;
    Real totR, r;
    bool requested;
    byte *inbuf, *outbuf;

    #ifndef MEGADEBUG
    MPI_Request request;
    #endif

    inline void unpackChunk(byte* buf, int & first, State& sOld, Action& a, Real& r, State& s);
    inline void packChunk(byte* buf, Action a);
    void save();

public:
    Master(Learner* const learner, Environment* env, Settings & settings);
    ~Master()
    {
        _dispose_object(env);
        _dispose_object(inbuf);
        _dispose_object(outbuf);
        _dispose_object(learner);
    }
    void run();
    void hustle();
    void restart(string fname);
};

class Slave
{
    Environment* const env;
    vector<Agent*> agents;
    const bool bTrain, bWriteToFile;
    int me, insize, outsize;
    byte *inbuf, *outbuf;

    vector<Action> actions;
    vector<State> States, oldStates;
    vector<int> info;
    string bufferTransition(const int iAgent) const;
    void packData(const int iAgent);
    void unpackData(const int iAgent);
    void sendFail(const int iAgent);
    void save() const;

public:
    Slave(Environment* env, int me, Settings & settings);
    ~Slave()
    {
        _dispose_object(env);
        _dispose_object(inbuf);
        _dispose_object(outbuf);
        for (auto & trash : agents) _dispose_object( trash);
    }
    void run();
    void restart(string fname);
    //Learner* learner; //TODO
};
