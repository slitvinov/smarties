//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Learners/AllLearners.h"
#include "Core/Master.h"

int main (int argc, char** argv)
{
  Settings settings;
  DistributionInfo distrib(argc, argv);

  CLI:App parser{"smarties : distributed reinforcement learning framework"};
  settings.initializeOpts(parser);
  distrib.initializeOpts(parser);
  try {
    app.parse(argc, argv);
  }
  catch (const CLI::ParseError &e) {
    if(distrib.world_rank == 0) return app.exit(e);
    else return 1;
  }
  MPI_Barrier(MPI_COMM_WORLD);

  distrib.initRandomSeed();
  distrib.figureOutWorkersPattern();
  settings.check();
  MPI_Barrier(MPI_COMM_WORLD);

  std::unique_ptr<Worker> process;

  if(distrib.bIsMaster)
  {
    if(distrib.nForkedProcesses2spawn > 0)
      process = std::make_unique<MasterSockets>(settings, distrib);
    else
      process = std::make_unique<MasterMPI>(settings, distrib);

    process->synchronizeEnvironments();
    process.setupCallers();
    process.runTraining();
  }
  else
  {
    process = std::make_unique<Worker>(settings, distrib);
    if(distrib.runInternalApp) // then worker lives inside the application
    {
      process->COMM.runApplication();
    }
    else
    {
      process->synchronizeEnvironments();
      comm_ptr->launch();
    }
  }

  return 0;
}
