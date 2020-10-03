#include "smarties.h"

extern "C" void app_main_(uintptr_t*, int*);
static int app_main_interface(smarties::Communicator *const smarties_comm,
                              MPI_Comm c_mpicomm, int argc, char **argv) {
  MPI_Fint f_mpicomm;
  uintptr_t i;
  f_mpicomm = MPI_Comm_c2f(c_mpicomm);
  i = (uintptr_t)(smarties_comm);
  app_main_(&i, &f_mpicomm);
  return 0;
}

int main(int argc, char **argv) {
  smarties::Engine e(argc, argv);
  if (e.parse())
    return 1;
  e.run(app_main_interface);
  return 0;
}
