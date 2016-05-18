/*
 *  ExternalEnvironment.h
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */


#pragma once

#include <vector>
#include "../Util/util.h"
#include "Environment.h"
#include "CellList.h"

//class ExternalAgent;
//#include "../Agents/ExternalAgent.h"

class ExternalEnvironment: public Environment
{
protected:
    const string execpath;
    const int rank;
    int n, iter;
    vector<int> ids;
    
    struct sockaddr_un serverAddress;
    struct sockaddr_un clientAddress;

    int workerid, callid, sock, ListenerSocket, bytes, probdim, servlen, sizein, sizeout;
    unsigned int addr_len;
    char SOCK_PATH[256];
    double *datain, *dataout;
    
    void spawn_server(int worker_id);
    
    //vector<Agent*> exagents;
    vector<State>  states;
    vector<Real> rewards;
    vector<Action> actions;

public:
    ExternalEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank);

    int  evolve(Real t) override;
    int  init  (      ) override;
    void close_Comm ( ) override;
    void setup_Comm ( ) override;
    virtual void setDims();
};


