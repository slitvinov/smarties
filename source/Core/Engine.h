//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#ifndef smarties_Engine_h
#define smarties_Engine_h

#include <memory>
#include "Settings.h"

namespace smarties
{

class Engine
{
  Settings settings; // todo will be vector and ini file parsing
  std::unique_ptr<DistributionInfo> distrib;

public:
  Engine(int argc, char** argv);
  Engine(MPI_Comm initialiazed_mpi_comm);

  void run();
  void init();
  int parse(int argc, char** argv);
};

} // end namespace smarties
#endif // smarties_Worker_h
