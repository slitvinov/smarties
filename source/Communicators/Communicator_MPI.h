//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Communicator_MPI_h
#define smarties_Communicator_MPI_h

#define SMARTIES_CORE

#include "Communicator.h"

// main function call for user's application
//  arguments are: - the communicator with smarties
//                 - the mpi communicator to use within the app
//                 - argc and argv read from settings file
int app_main(smarties::Communicator*const smartiesCommunicator,
             const MPI_Comm mpiCommunicator,
             int argc, char**argv);

#endif // smarties_Communicator_MPI_h
