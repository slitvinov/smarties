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
#include "../Communicator.h"
#include <map>

class Builder;

class Environment
{
protected:
    mt19937 * const g; //only ok if only thread 0 accesses
    Settings & settings;
    Communicator* comm_ptr = nullptr;
    void commonSetup();

public:
    Uint nAgents, nAgentsPerRank;
    const Real gamma;

    Uint mpi_ranks_per_env = 0;
    string paramsfile = string();

    vector<Agent*> agents;
    StateInfo  sI;
    ActionInfo aI;
    Environment(Settings & _settings);

    virtual ~Environment();

    virtual void setDims ();

    virtual bool pickReward(const Agent& agent);
    virtual bool predefinedNetwork(Builder* const net) const;
    Communicator create_communicator( const MPI_Comm slavesComm, const int socket, const bool bSpawn);

    virtual Uint getNdumpPoints();
    virtual vector<Real> getDumpState(Uint k);
};
