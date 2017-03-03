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
    const MPI_Comm slavesComm;
    Learner* const learner;
    Environment* const env;
    const ActionInfo aI;
    const StateInfo  sI;
    const vector<Agent*> agents;
    const int bTrain, nAgents, nSlaves, saveFreq, inOneSize, outOneSize;
    byte *const inbuf;
    byte *const outbuf;
    mt19937 * const gen;
    State  sOld, sNew;
    Action aOld, aNew;
    Real totR;
    bool requested;
    int iter;

    MPI_Request request;

    inline void unpackChunk(byte* buf, int & first, State& sOld, Action& a, Real& r, State& s);
    inline void packChunk(byte* buf, Action a);
    void save();

    byte * _alloc(const int size) {
      return new byte[size];
    }
    void _dealloc(byte* ptr) {
        if(ptr not_eq nullptr) {
            delete [] ptr;
            ptr=nullptr;
        }
    }

public:
    Master(MPI_Comm comm, Learner*const learner, Environment*const env, Settings& settings);
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
    const MPI_Comm slavesComm;
    Environment* const env;

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
    Slave(MPI_Comm comm, Environment * const env, int me, Settings & settings);
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

class Client
{
  private:
      Learner* const learner;
      Environment* const env;
      vector<Agent*> agents;
      ActionInfo actInfo;
      StateInfo  sInfo;
      State  sOld, s;
      Action aOld, a;
      Real r;

  public:
      Client(Learner*const learner, Environment*const env, Settings& settings);
      ~Client()
      {
          _dispose_object(env);
          _dispose_object(learner);
      }
};
