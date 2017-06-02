/*
 *  Environment.h
 *  rl
 *
 *  Created by Guido Novati on 21.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "../Agent.h"
#include "../Network/Builder.h"
#include "../Communicator.h"
#include <map>

class Environment
{
protected:
    mt19937 * const g; //only ok if only thread 0 accesses
    Settings & settings;
    Communicator* comm_ptr = nullptr;
    void commonSetup();

public:
    const string execpath;
    const Uint nAgents, nAgentsPerRank;
    const double gamma;

    bool resetAll = false;
    bool cheaperThanNetwork = true;
    Uint mpi_ranks_per_env = 0;
    string paramsfile = string();

    vector<Agent*> agents;
    StateInfo  sI;
    ActionInfo aI;
    Environment(const Uint nAgents, const string execpath, Settings & settings);

    virtual ~Environment();

    virtual void setDims () = 0;

    virtual bool pickReward(const State& t_sO, const Action& t_a,
                            const State& t_sN, Real& reward, const int info);
    virtual bool predefinedNetwork(Builder* const net) const;
    Communicator create_communicator( const MPI_Comm slavesComm, const int socket, const bool bSpawn);

    virtual vector<Real> stateDumpUpperBound();
    virtual vector<Real> stateDumpLowerBound();
    virtual vector<Uint> stateDumpNBins();
};
