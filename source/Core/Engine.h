//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#ifndef smarties_Engine_h
#define smarties_Engine_h

#include "../Communicators/Communicator.h"


namespace smarties
{

struct DistributionInfo;

#define VISIBLE __attribute__((visibility("default")))

class Engine
{
  DistributionInfo * const distrib;

  void init();

public:
  VISIBLE Engine(int argc, char** argv);

  VISIBLE Engine(MPI_Comm initialiazed_mpi_comm, int argc, char** argv);

  VISIBLE ~Engine();

  VISIBLE void run(const environment_callback_t & callback);

  VISIBLE void run(const environment_callback_MPI_t & callback);

  VISIBLE int parse();

  VISIBLE void setNthreads(const Uint nThreads);

  VISIBLE void setNmasters(const Uint nMasters);

  VISIBLE void setNworkers(const Uint nWorkers);

  VISIBLE void setNworkersPerEnvironment(const Uint workerProcessesPerEnv);

  VISIBLE void setRandSeed(const Uint randSeed);

  VISIBLE void setTotNumTimeSteps(const Uint totNumSteps);

  VISIBLE void setSimulationArgumentsFilePath(const std::string& appSettings);

  VISIBLE void setSimulationSetupFolderPath(const std::string& setupFolder);

  VISIBLE void setRestartFolderPath(const std::string& restart);

  VISIBLE void setIsTraining(const bool bTrain);

  VISIBLE void setIsLoggingAllData(const bool logAllSamples);

  VISIBLE void setAreLearnersOnWorkers(const bool learnersOnWorkers);
};

#undef VISIBLE

} // end namespace smarties
#endif // smarties_Engine_h
