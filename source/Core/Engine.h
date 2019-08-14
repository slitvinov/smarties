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
  Engine(int argc, char** argv);
  Engine(MPI_Comm initialiazed_mpi_comm, int argc, char** argv);
  ~Engine();

  void run(const environment_callback_t & callback);
  void run(const environment_callback_MPI_t & callback);

  int parse();
};

} // end namespace smarties
#endif // smarties_Engine_h
