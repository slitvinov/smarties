//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_MPIUtilities_h
#define smarties_MPIUtilities_h

#ifndef SMARTIES_LIB
#include <mpi.h>
#include <omp.h>
#endif

namespace smarties
{

#ifdef MPI_VERSION

inline MPI_Comm MPICommDup(const MPI_Comm C) {
  MPI_Comm ret;
  MPI_Comm_dup(C, &ret);
  return ret;
}
inline unsigned MPICommSize(const MPI_Comm C) {
  int size;
  MPI_Comm_size(C, &size);
  return (unsigned) size;
}
inline unsigned MPICommRank(const MPI_Comm C) {
  int rank;
  MPI_Comm_rank(C, &rank);
  return (unsigned) rank;
}
inline unsigned MPIworldRank() { return MPICommRank(MPI_COMM_WORLD); }

#else

#ifndef SMARTIES_LIB
  #error "mpi.h did not define MPI_VERSION"
#endif

inline unsigned MPIworldRank() { return 0; }

#endif

#define MPI(NAME, ...)                                   \
do {                                                     \
  int MPIERR = 0;                                        \
  if(distrib.bAsyncMPI) {                                \
    MPIERR = MPI_ ## NAME ( __VA_ARGS__ );               \
  } else {                                               \
    std::lock_guard<std::mutex> lock(distrib.mpiMutex);  \
    MPIERR = MPI_ ## NAME ( __VA_ARGS__ );               \
  }                                                      \
  if(MPIERR not_eq MPI_SUCCESS) {                        \
    _warn("%s %d", #NAME, MPIERR);                       \
    throw std::runtime_error("MPI ERROR");               \
  }                                                      \
} while(0)
/*
#define MPI(NAME, ...)                                   \
do {                                                     \
  int MPIERR = MPI_ ## NAME ( __VA_ARGS__ );             \
  if(MPIERR not_eq MPI_SUCCESS) {                        \
    _warn("%s %d", #NAME, MPIERR);                       \
    throw std::runtime_error("MPI ERROR");               \
  }                                                      \
} while(0)
*/

} // end namespace smarties
#endif // smarties_MPIUtilities_h
