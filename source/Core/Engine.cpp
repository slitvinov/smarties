//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Engine.h"
#include "Master.h"

namespace smarties
{

Engine::Engine(int argc, char** argv) :
  distrib(new DistributionInfo(argc, argv)) { }

Engine::Engine(MPI_Comm mpi_comm, int argc, char** argv) :
  distrib(new DistributionInfo(mpi_comm, argc, argv)) { }

Engine::~Engine()
{
  assert(distrib not_eq nullptr);
  delete distrib;
}

int Engine::parse()
{
  return distrib->parse();
}

void Engine::init()
{
  distrib->initialzePRNG();
  distrib->figureOutWorkersPattern();

  if(distrib->bTrain == false && distrib->restart == "none") {
   printf("Did not specify path for restart files, assumed current dir.\n");
   distrib->restart = ".";
  }

  MPI_Barrier(distrib->world_comm);
}

void Engine::run(const environment_callback_t & callback)
{
  init();

  if(distrib->bIsMaster)
  {
    if(distrib->nForkedProcesses2spawn > 0) {
      MasterSockets process(*distrib);
      process.run(callback);
    } else {
      MasterMPI     process(*distrib);
      process.run();
    }
  }
  else
  {
    Worker          process(*distrib);
    process.run(callback);
  }
}

void Engine::run(const environment_callback_MPI_t & callback)
{
  init();

  if(distrib->bIsMaster)
  {
    assert(distrib->nForkedProcesses2spawn <= 0);
    MasterMPI process(*distrib);
    process.run();
  }
  else
  {
    Worker    process(*distrib);
    process.run(callback);
  }
}

}
