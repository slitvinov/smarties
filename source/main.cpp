//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Core/Master.h"
#include "Settings.h"
#include "CLI/CLI.hpp"

int main (int argc, char** argv)
{
  const auto settings = std::make_unique<smarties::Settings>();
  const auto distrib = std::make_unique<smarties::DistributionInfo>(argc, argv);

  CLI::App parser("smarties : distributed reinforcement learning framework");
  settings->initializeOpts(parser);
  distrib->initializeOpts(parser);
  try {
    parser.parse(argc, argv);
  }
  catch (const CLI::ParseError &e) {
    if(distrib->world_rank == 0) return parser.exit(e);
    else return 1;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  distrib->initialzePRNG();
  distrib->figureOutWorkersPattern();
  settings->check();
  MPI_Barrier(MPI_COMM_WORLD);

  std::shared_ptr<smarties::Worker> process;

  if(distrib->bIsMaster)
  {
    if(distrib->nForkedProcesses2spawn > 0)
      process = std::make_shared<smarties::MasterSockets>(* settings.get(),
                                                          *  distrib.get() );
    else
      process = std::make_shared<smarties::MasterMPI>(* settings.get(),
                                                      *  distrib.get() );
  }
  else
  {
    process = std::make_shared<smarties::Worker>(* settings.get(),
                                                 *  distrib.get() );
  }
  process->run();
  return 0;
}
