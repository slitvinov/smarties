/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "Environment.h"

using namespace std;

Environment::Environment(const int nA,const string exe,const int _rank,Settings& _s) :
execpath(exe), rank(_rank), nAgents(nA*_s.nSlaves), nAgentsPerRank(nA),
gamma(_s.gamma), g(&_s.generators[0]), resetAll(false), cheaperThanNetwork(true),
mpi_ranks_per_env(0), paramsfile(string())
{
    assert(_s.bIsMaster || nAgentsPerRank == nAgents);
    for (int i=0; i<nAgents; i++) agents.push_back(new Agent(i));
}

Environment::~Environment()
{
    for (auto & trash : agents)
      _dispose_object(trash);
}

bool Environment::predefinedNetwork(Network* const net) const
{
	//this function can be used if environment requires particular network settings
	//i.e. not fully connected LSTM/FF network
	//i.e. if you want to use convolutions
	return false;
}

void Environment::commonSetup()
{
    int wRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &wRank);
    sI.dim = 0; sI.dimUsed = 0;
    for (int i=0; i<sI.inUse.size(); i++) {
        sI.dim++;
        if (sI.inUse[i]) sI.dimUsed++;
    }
    if(!wRank)
    printf("State has %d component, %d in use\n", sI.dim, sI.dimUsed);

    aI.updateShifts();

    if(! aI.bounded.size()) {
      aI.bounded.resize(aI.dim, 0);
      if(!wRank)
      printf("Unspecified whether action space is bounded: assumed not\n");
    } else assert(aI.bounded.size() == aI.dim);

    for (auto& a : agents) {
        a->setDims(sI, aI);
        a->a = new Action(aI, g);
        a->s = new State(sI);
        a->sOld = new State(sI);
    }
}

bool Environment::pickReward(const State& t_sO, const Action& t_a,
                             const State& t_sN, Real& reward, const int info)
{
    return info == 2;
}

void Environment::stateBounds(vector<Real>& lower, vector<Real>& upper, vector<int>& nbins)
{
  lower = vector<Real>(0);
  upper = vector<Real>(0);
  nbins = vector<int> (0);
}
/*
 void GlideEnvironment::setDims()
 {
 sI.dim = 6;
 // State: u velocity...
 sI.bounds.push_back(7);
 sI.top.push_back(1);
 sI.bottom.push_back(0);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);

 // ...v velocity...
 sI.bounds.push_back(7);
 sI.top.push_back(0);
 sI.bottom.push_back(-1);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);

 // ...angular velocity...
 sI.bounds.push_back(7);
 sI.top.push_back(0.5);
 sI.bottom.push_back(-0.5);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);

 // ...x pos...
 sI.bounds.push_back(26);
 sI.top.push_back(20);
 sI.bottom.push_back(-100);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);

 // ...y pos...
 sI.bounds.push_back(20);
 sI.top.push_back(50.001);
 sI.bottom.push_back(-.001);
 sI.aboveTop.push_back(false);
 sI.belowBottom.push_back(false);

 // ...angle...
 sI.bounds.push_back(12);
 sI.top.push_back(3.14159265359);
 sI.bottom.push_back(-3.14159265359);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);

 // ...torque...
 //sI.bounds.push_back(10);
 //sI.top.push_back(0.8);
 //sI.bottom.push_back(-0.8);
 //sI.aboveTop.push_back(true);
 //sI.belowBottom.push_back(true);

aI.dim = 1;

for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
}
*/
