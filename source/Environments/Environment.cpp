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

Environment::Environment(const int _nAgents, const string _execpath, const int _rank, Settings & settings) :
execpath(_execpath), rank(_rank), isLauncher(settings.isLauncher), nAgents(_nAgents),
gamma(settings.gamma), g(settings.gen), resetAll(true), iter(0), communicator(nullptr), workid(settings.sockPrefix)
{
    for (int i=0; i<nAgents; i++) agents.push_back(new Agent(i));
}

void Environment::setup_Comm()
{
    if(communicator == nullptr)
      communicator = new Communicator(workid,sI.dim,aI.dim,isLauncher==1,false);
      //die("Set up the state before establishing a communication.\n");

    if (isLauncher)
      communicator->setupClient(iter, execpath);
    else
      communicator->setupServer();
}

Environment::~Environment()
{
    _dispose_object(communicator);
    for (auto & trash : agents)
      _dispose_object(trash);
}

void Environment::close_Comm()
{
    communicator->closeSocket();
}

void Environment::setAction(const int& iAgent)
{
    communicator->sendAction(agents[iAgent]->a->vals);
}

int Environment::getState(int& iAgent)
{
    _AGENT_STATUS status;
    vector<double> state(sI.dim);
    double reward;
    communicator->recvState(iAgent, status, state, reward);

    assert(iAgent<nAgents);
    std::swap(agents[iAgent]->s,agents[iAgent]->sOld);
    for (int j=0; j<sI.dim; j++)
    agents[iAgent]->s->vals[j] = state[j];
    agents[iAgent]->r = reward;
    agents[iAgent]->Status = status;
    return status;
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
    sI.dim = 0; sI.dimUsed = 0;
    for (int i=0; i<sI.inUse.size(); i++) {
        sI.dim++;
        if (sI.inUse[i]) sI.dimUsed++;
    }
    printf("State has %d component, %d in use\n", sI.dim, sI.dimUsed);

    aI.shifts.resize(aI.dim);
    aI.shifts[0] = 1;
    for (int i=1; i < aI.dim; i++) {
        assert(aI.bounds[i] == aI.values[i].size());
        aI.shifts[i] = aI.shifts[i-1] * aI.bounds[i-1];
    }

    if(! aI.bounded.size()) {
      aI.bounded.resize(aI.dim, 0);
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
