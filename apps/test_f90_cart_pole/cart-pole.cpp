//
//  cart-pole
//
//  Created by Jacopo Canton on 01/02/19.
//  Copyright (c) 2019 Jacopo Canton. All rights reserved.
//
#include <iostream>
#include <vector>

#include "mpi.h"
#include "Communicators/Communicator.h"
//=============================================================================


extern "C" void rlcomm_set_state_observable(
    void* opaque_ptr_to_class,
    bool *observable,
    int array_len
) {
  std::vector<bool> observable_vec;

  // Cast to Communicator (class)
  Communicator *ptr_to_class = static_cast<Communicator*>(opaque_ptr_to_class);
  // Assign the vector
  observable_vec.assign(observable, observable+array_len);

  // Call the public member function
  ptr_to_class->set_state_observable(observable_vec);
}

//=============================================================================


//=============================================================================
// Entry point into fortran code.  Fortran defines this.
extern "C" void fortran_app_main(const void* rlcomm, const int f_mpicomm);
//=============================================================================


//=============================================================================
// Program entry point
int app_main(
  Communicator*const rlcomm, // communicator with smarties
  MPI_Comm c_mpicomm,        // mpi_comm that mpi-based apps can use (C handle)
  int argc, char**argv,      // arguments read from app's runtime settings file
  const unsigned numSteps    // number of time steps to run before exit
) {
  std::cout << "C++ side begins" << std::endl;

  // Convert the C handle to the MPI communicator to a Fortran handle
  MPI_Fint f_mpicomm;
  f_mpicomm = MPI_Comm_c2f(c_mpicomm);

  fortran_app_main(rlcomm, f_mpicomm);

  std::cout << "C++ side ends" << std::endl;
  return 0;
} // main
//=============================================================================
