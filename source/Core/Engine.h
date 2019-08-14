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

class Engine
{
  DistributionInfo * const distrib;

  void init();

public:
  __attribute__((visibility("default")))
  Engine(int argc, char** argv);

  __attribute__((visibility("default")))
  Engine(MPI_Comm initialiazed_mpi_comm, int argc, char** argv);

  __attribute__((visibility("default")))
  ~Engine();

  __attribute__((visibility("default")))
  void run(const environment_callback_t & callback);

  __attribute__((visibility("default")))
  void run(const environment_callback_MPI_t & callback);

  __attribute__((visibility("default")))
  int parse();
};

} // end namespace smarties
#endif // smarties_Engine_h
