//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Core/Engine.h"
#include "Core/Master.h"
//#include "json.hpp"
#include "CLI/CLI.hpp"

namespace smarties
{

//using json = nlohmann::json;

Engine::Engine(int argc, char** argv)
{
  distrib = std::make_unique<DistributionInfo>(argc, argv);
}

Engine::Engine(MPI_Comm mpi_comm, int argc, char** argv)
{
  distrib = std::make_unique<DistributionInfo>(mpi_comm, argc, argv);
  // TODO read json
}

int Engine::parse()
{
  CLI::App parser("smarties : distributed reinforcement learning framework");
  settings.initializeOpts(parser);
  distrib->initializeOpts(parser);
  try {
    parser.parse(distrib->argc, distrib->argv);
  }
  catch (const CLI::ParseError &e) {
    if(distrib->world_rank == 0) return parser.exit(e);
    else return 1;
  }
  MPI_Barrier(distrib->world_comm);
  return 0;
}

void Engine::init()
{
  distrib->initialzePRNG();
  distrib->figureOutWorkersPattern();
  settings.defineDistributedLearning(*distrib.get());
  settings.check();
  MPI_Barrier(distrib->world_comm);
}

void Engine::run(const environment_callback_t & callback)
{
  init();

  if(distrib->bIsMaster)
  {
    if(distrib->nForkedProcesses2spawn > 0) {
      MasterSockets process(settings, *distrib.get());
      process.run(callback);
    } else {
      MasterMPI     process(settings, *distrib.get());
      process.run();
    }
  }
  else
  {
    Worker          process(settings, *distrib.get());
    process.run(callback);
  }
}

void Engine::run(const environment_callback_MPI_t & callback)
{
  init();

  if(distrib->bIsMaster)
  {
    assert(distrib->nForkedProcesses2spawn <= 0);
    MasterMPI process(settings, *distrib.get());
    process.run();
  }
  else
  {
    Worker    process(settings, *distrib.get());
    process.run(callback);
  }
}

}
