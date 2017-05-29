//
//  Scheduler.h
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#pragma once

class Learner;
#include "Communicator.h"
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
    const int bTrain, nPerRank, nSlaves, nThreads, saveFreq, inSize, outSize;
    double*const inbuf;
    double*const outbuf;
    State  sOld, sNew;
    Action aOld, aNew;
    Real meanR = 0, varR = 0;
    unsigned long iter = 0;
    vector<int> status;

    MPI_Request request;

    void recvState(const int slave, int& iAgent, int& istatus, Real& reward);
    void sendAction(const int slave, const int iAgent);
    void save();
/*
    double * _alloc(const int size) {
      //return new byte[size];
      double* ret = (double*) malloc(size);
      memset(ret, 0, size);
      return ret;
    }
    void _dealloc(double* ptr) {
        if(ptr not_eq nullptr) {
            //delete [] ptr;
            free(ptr);
            ptr=nullptr;
        }
    }
*/
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
    void restart(string fname);
};

class Slave
{
  private:
    Communicator* const comm;
    Environment* const env;
    const bool bTrain;
    vector<int> status;

  public:
    Slave(Communicator*const c, Environment*const e, Settings& s);
    ~Slave()
    {
        _dispose_object(env);
    }
    void run();
};

class Client
{
  private:
    Learner* const learner;
    Communicator* const comm;
    Environment* const env;
    vector<Agent*> agents;
    const ActionInfo aI;
    const StateInfo  sI;
    State  sOld, sNew;
    Action aOld, aNew;
    vector<int> status;
    void prepareState(int& iAgent, int& istatus, Real& reward);
    void prepareAction(const int iAgent);

  public:
    Client(Learner*const l,Communicator*const c,Environment*const e,Settings&s);
    ~Client()
    {
        _dispose_object(env);
        _dispose_object(learner);
    }
    void run();
};
