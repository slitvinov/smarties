/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "Environment.h"
#include "../Network/Builder.h"
#ifndef GYM_RENDEROPT
#define GYM_RENDEROPT 0
#endif

Environment::Environment(Settings& _settings) :
g(&_settings.generators[0]), settings(_settings), gamma(_settings.gamma) {}

void Environment::setDims() //this environment is for the cart pole test
{
  #if   GYM_RENDEROPT==0
    comm_ptr->dump_value = settings.bTrain||settings.slaves_rank>1 ? -1 : 1;
  #elif GYM_RENDEROPT==1
    comm_ptr->dump_value = settings.slaves_rank>1 ? -1 : 1;
  #elif GYM_RENDEROPT==2
    comm_ptr->dump_value = settings.slaves_rank>1 ? -1 : 2;
  #else
    comm_ptr->dump_value = 1;
  #endif

  comm_ptr->getStateActionShape();
  assert(comm_ptr->nAgents>0);
  assert(comm_ptr->nStates>0);
  assert(comm_ptr->nActions>0);
  assert(comm_ptr->discrete_actions>=0);

  sI.dim = comm_ptr->nStates;
  aI.dim = comm_ptr->nActions;
  nAgentsPerRank = comm_ptr->nAgents;
  aI.discrete = comm_ptr->discrete_actions;

  aI.values.resize(aI.dim);
  aI.bounded.resize(aI.dim, 0);
  sI.mean.resize(sI.dim);
  sI.scale.resize(sI.dim);
  sI.inUse.resize(sI.dim, 1);

  if(!settings.world_rank) printf("State dimensionality : %d.",sI.dim);
  for (unsigned i=0; i<sI.dim; i++) {
    const bool inuse = comm_ptr->obs_inuse[i] > 0.5;
    const double upper = comm_ptr->obs_bounds[i*2+0];
    const double lower = comm_ptr->obs_bounds[i*2+1];
    sI.mean[i]  = 0.5*(upper+lower); sI.inUse[i] = inuse;
    sI.scale[i] = 0.5*std::fabs(upper-lower)/std::sqrt(3.); //approximate std=1
    if(sI.scale[i]>=1e3 || sI.scale[i] < 1e-7) {
      if(!settings.world_rank) printf(" unbounded");
      sI.scale = Rvec(); sI.mean = Rvec();
      break;
    }
    if(!settings.world_rank && not inuse)
      printf(" State component %u is hidden from the learner.", i);
  }
  if(!settings.world_rank) printf("\nAction dimensionality : %d.",aI.dim);

  int k = 0;
  for (Uint i=0; i<aI.dim; i++) {
    aI.bounded[i] = comm_ptr->action_options[i*2+1] > 0.5;
    const int nvals = comm_ptr->action_options[i*2];
    assert(aI.discrete || nvals == 2);
    assert(nvals > 1);
    aI.values[i].resize(nvals);
    for(int j=0; j<nvals; j++)
      aI.values[i][j] = comm_ptr->action_bounds[k++];

    const Real amax = aI.getActMaxVal(i), amin = aI.getActMinVal(i);
    if(!settings.world_rank)
    printf(" [%u: %f:%f%s]", i, amin, amax, aI.bounded[i]?" (bounded)":"");
  }
  if(!settings.world_rank) printf("\n");

  commonSetup(); //required
  assert(sI.dim == (Uint) comm_ptr->nStates);
}

Communicator Environment::create_communicator(
  const MPI_Comm slavesComm,
  const int socket,
  const bool bSpawn)
{
  assert(socket>0);
  Communicator comm(slavesComm, socket, bSpawn);
  comm.set_exec_path(settings.launchfile);
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

bool Environment::predefinedNetwork(Builder & input_net) const
{
  // this function is to be filled by child classes
  // to implement convolutional models
  return false;
}

void Environment::commonSetup()
{
  assert(settings.nSlaves > 0);
  assert(nAgentsPerRank > 0);
  nAgents = nAgentsPerRank * settings.nSlaves;
  agents.resize(std::max(nAgents, (Uint) 1), nullptr);
  for(Uint i=0; i<nAgents; i++) agents[i] = new Agent(i);
  settings.nAgents = nAgents;

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
  for (Uint i=0; i<sI.scale.size(); i++) assert(positive(sI.scale[i]));
}

bool Environment::pickReward(const Agent& agent)
{
  return agent.Status == 2;
}

Uint Environment::getNdumpPoints()
{
  return 0;
}

Rvec Environment::getDumpState(Uint k)
{
  return Rvec();
}
