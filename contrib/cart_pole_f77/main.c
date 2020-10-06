#include <mpi.h>
#include <stdint.h>
#include <smarties_f77.h>

void app_main_(uintptr_t*, int *);
static int main0(uintptr_t *smarties, void *mpi0, void *p) {
  MPI_Fint f_mpicomm;
  MPI_Comm *mpi;
  mpi = (MPI_Comm*)mpi0;
  f_mpicomm = MPI_Comm_c2f(*mpi);
  app_main_(smarties, &f_mpicomm);
  return 0;
}

int main(int argc, char **argv) {
    void *p;
    return smarties_main(argc, argv, main0, p);
}
