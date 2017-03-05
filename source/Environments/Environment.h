/*
 *  Environment.h
 *  rl
 *
 *  Created by Guido Novati on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

//#include "../Util/util.h"
#include <sys/un.h>
#include "../Util/Communicator.h"
#include "../Agent.h"
#include "../StateAction.h"
#include "../ANN/Network.h"

#include <map>

class Environment
{
protected:
    mt19937 * g;
    Communicator * communicator;

    void commonSetup();

public:
    const string execpath;
    const int rank, isLauncher, nAgents, nAgentsPerRank, workid;
    const double gamma;
    long unsigned int iter;
    bool resetAll;
    vector<Agent*> agents;
    StateInfo sI;
    ActionInfo aI;
    Environment(const int nAgents, const string execpath,
                const int _rank, Settings & settings);

    virtual ~Environment();

    virtual void setDims () = 0;
    virtual int getState(int & iAgent) ;
    virtual void setAction(const int & iAgent);
    void close_Comm ();
    void setup_Comm ();
    virtual bool pickReward(const State& t_sO, const Action& t_a,
                            const State& t_sN, Real& reward, const int info);
    virtual bool predefinedNetwork(Network* const net) const;
};
