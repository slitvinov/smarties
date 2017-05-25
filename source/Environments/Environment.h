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

#include <map>

class Environment
{
protected:
    mt19937 * const g; //only ok if only thread 0 accesses
    void commonSetup();

public:
    const string execpath;
    string paramsfile;
    const int rank, nAgents, nAgentsPerRank;
    const double gamma;
    bool resetAll, cheaperThanNetwork;
    int mpi_ranks_per_env;
    vector<Agent*> agents;
    StateInfo  sI;
    ActionInfo aI;
    Environment(const int nAgents, const string execpath,
                const int _rank, Settings & settings);

    virtual ~Environment();

    virtual void setDims () = 0;

    virtual bool pickReward(const State& t_sO, const Action& t_a,
                            const State& t_sN, Real& reward, const int info);
    virtual bool predefinedNetwork(Builder* const net) const;

    virtual vector<Real> stateDumpUpperBound();
    virtual vector<Real> stateDumpLowerBound();
    virtual vector<int> stateDumpNBins();
};
