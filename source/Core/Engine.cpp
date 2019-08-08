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

Engine::Engine(MPI_Comm initialiazed_mpi_comm)
{
  distrib = std::make_unique<DistributionInfo>(initialiazed_mpi_comm);
  // TODO read json
}

int Engine::parse(int argc, char** argv)
{
  CLI::App parser("smarties : distributed reinforcement learning framework");
  settings.initializeOpts(parser);
  distrib->initializeOpts(parser);
  try {
    parser.parse(argc, argv);
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

void Engine::run()
{
  std::shared_ptr<Worker> process;

  if(distrib->bIsMaster)
  {
    if(distrib->nForkedProcesses2spawn > 0)
      process = std::make_shared<MasterSockets>(settings, *distrib.get());
    else
      process = std::make_shared<MasterMPI>(settings, *distrib.get());
  }
  else
  {
    process = std::make_shared<Worker>(settings, *distrib.get());
  }
  process->run();
}

}
