/*
 *  Environment.h
 *  rl
 *
 *  Created by Guido Novati on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#include "../Util/util.h"
#include "../Agent.h"
#include "../StateAction.h"
#include "../ANN/Network.h"

#include <map>

class Environment
{
protected:
    const string execpath;
    const int rank;
    mt19937 * g;
    int nAgents, workerid, sock, ListenerSocket, bytes, sizein, sizeout;
    const double gamma;
    
    struct sockaddr_un serverAddress;
    struct sockaddr_un clientAddress;
    char SOCK_PATH[256];
    double *datain, *dataout;
    void commonSetup();
    virtual void spawn_server();
public:
    long unsigned int iter;
    bool resetAll;
    vector<Agent*> agents;
    vector<Real> max_scale, min_scale;
    StateInfo sI;
	ActionInfo aI;


    Environment(const int nAgents, const string execpath, const int _rank, Settings & settings);
    
    virtual ~Environment()
    {
        _dispose_object(datain);
        _dispose_object(dataout);
        for (auto & trash : agents)
        	_dispose_object( trash);
    }
    
    virtual void setDims () = 0;
    virtual int getState(int & iAgent) ;
    virtual void setAction(const int & iAgent);
    void close_Comm ();
    void setup_Comm ();
    virtual bool pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward, const int info) = 0;
	virtual bool predefinedNetwork(Network* const net) const;
};
