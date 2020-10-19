#include <stdint.h>
#include <smarties_f77.h>
#include <stdio.h>
#include <mpi.h>

extern int app_main_(uintptr_t *, int *);
static int main0(uintptr_t *smarties, void *mpi0, void *p) {
  MPI_Fint f_mpicomm;
  MPI_Comm *mpi;
  FILE *file;

  mpi = (MPI_Comm *)mpi0;
  f_mpicomm = MPI_Comm_c2f(*mpi);
  return app_main_(smarties, &f_mpicomm);
}

int main(int argc, char **argv) {
  return smarties_main_(argc, argv, main0, NULL);
}
