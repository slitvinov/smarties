#ifdef __Smarties_
#undef __Smarties_
#endif
#include "mpi.h"
#include "Communicator.h"

#include <iostream>
#include <cmath>
#include <cassert>

#include <random>

int app_main(Communicator*const rlcom, MPI_Comm mpicom, int argc, char**argv)
{
  for (int i = 0; i < argc; ++i)
      std::cout << argv[i] << std::endl;

  int rank, size;
  std::mt19937 gen(0);
  std::normal_distribution<double> dist(0, 1);
  MPI_Comm_rank(mpicom, &rank);
  MPI_Comm_size(mpicom, &size);
  std::cout << rank << " " << size << std::endl;
  std::vector<double> state(rlcom->getStateDim());
  std::vector<double> action(rlcom->getActionDim());

  while(true) {
    int status = 1;
    for(int k=0; k<10; k++)
    {
      if(k==9) status = 2;

      for(int i=0; i<rlcom->getStateDim(); i++)
        state[i] = dist(gen);
      double r = dist(gen);

      rlcom->sendState(0, status, state, r);
      if(status != 2)
      rlcom->recvAction(action);
      status = 0;
    }
  }
  return 0;
}
