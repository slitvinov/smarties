//==============================================================================
//
// main.cpp
// Part of the cart_pole_f90 example.
//
// This is the starting point.
//
// Copyright (c) 2019 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
// Distributed under the terms of the MIT license.
//
//=============================================================================
#include <iostream>
#include <cstdio>
#include "smarties.h"


//==============================================================================
// Declaration of the entry point into Fortran code
extern "C" void app_main(const void* smarties_comm, const int f_mpicomm);


//==============================================================================
// Interface to the Fortran app_main
int app_main_interface(
  smarties::Communicator*const smarties_comm, // communicator with smarties
  MPI_Comm c_mpicomm,   // mpi_comm that mpi-based apps can use (C handle)
  int argc, char**argv  // arguments read from app's runtime settings file
)
{
  // Convert the C handle to the MPI communicator to a Fortran handle
  MPI_Fint f_mpicomm;
  f_mpicomm = MPI_Comm_c2f(c_mpicomm);

  // Call the main app for the training (in Fortran)
  app_main(smarties_comm, f_mpicomm);
  return 0;
}


//==============================================================================
// Main program
int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main_interface );
  return 0;
}
//==============================================================================
