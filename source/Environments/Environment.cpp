/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "Environment.h"

Environment::Environment(const Uint nA, const string exe, Settings& _s) :
execpath(exe), nAgents(nA*_s.nSlaves), nAgentsPerRank(nA),
gamma(_s.gamma), g(&_s.generators[0]), settings(_s)
{
    assert(!_s.slaves_rank || nAgentsPerRank == nAgents);
    for (Uint i=0; i<nAgents; i++) agents.push_back(new Agent(i));
    _s.nAgents = agents.size();
}

Communicator Environment::create_communicator(
  const MPI_Comm slavesComm,
  const int socket,
  const bool bSpawn)
{
  assert(socket>0);
  Communicator comm(slavesComm, socket, bSpawn);
  comm.set_exec_path(execpath);
  comm_ptr = &comm;
  if(mpi_ranks_per_env==0 && settings.slaves_rank>0)
    comm_ptr->launch();
  setDims();
  comm.update_state_action_dims(sI.dim, aI.dim);

  if(mpi_ranks_per_env>0 && settings.slaves_rank>0)
  {
    assert(bSpawn);
    assert(paramsfile != string());
    settings.nSlaves = settings.slaves_size-1; //one is the master

    if(settings.nSlaves % mpi_ranks_per_env != 0)
			die("Number of ranks does not match app\n");

    int slaveGroup = (settings.slaves_rank-1) / mpi_ranks_per_env;
		MPI_Comm app_com;
		MPI_Comm_split(slavesComm, slaveGroup, settings.slaves_rank, &app_com);

    comm.set_params_file(paramsfile);
    comm.set_application_mpicom(app_com, slaveGroup);
    while(true)
      comm.ext_app_run();
  }

  return comm;
}
Environment::~Environment()
{
    for (auto & trash : agents)
      _dispose_object(trash);
}

bool Environment::predefinedNetwork(Builder* const net) const
{
	//this function can be used if environment requires particular network settings
	//i.e. not fully connected LSTM/FF network
	//i.e. if you want to use convolutions
	return false;
}

void Environment::commonSetup()
{
    sI.dim = 0; sI.dimUsed = 0;
    for (Uint i=0; i<sI.inUse.size(); i++) {
        sI.dim++;
        if (sI.inUse[i]) sI.dimUsed++;
    }
    if(!settings.world_rank)
    printf("State has %d component, %d in use\n", sI.dim, sI.dimUsed);

    aI.updateShifts();

    if(! aI.bounded.size()) {
      aI.bounded.resize(aI.dim, 0);
      if(!settings.world_rank)
      printf("Unspecified whether action space is bounded: assumed not\n");
    } else assert(aI.bounded.size() == aI.dim);

    for (auto& a : agents) {
        a->setDims(sI, aI);
        a->a = new Action(aI, g);
        a->s = new State(sI);
        a->sOld = new State(sI);
    }
    assert(sI.scale.size() == sI.mean.size());
    assert(sI.mean.size()==0 || sI.mean.size()==sI.dim);
    for (Uint i=0; i<sI.dim; i++) assert(positive(sI.scale[i]));
}

bool Environment::pickReward(const State& t_sO, const Action& t_a,
                             const State& t_sN, Real& reward, const int info)
{
    return info == 2;
}

vector<Real> Environment::stateDumpUpperBound() {return vector<Real>(0);}
vector<Real> Environment::stateDumpLowerBound() {return vector<Real>(0);}
vector<Uint> Environment::stateDumpNBins() {return vector<Uint>(0);}
