#include "smarties.h"

extern "C" void app_main_(void *smarties_comm, int f_mpicomm);
static int app_main_interface(smarties::Communicator *const smarties_comm,
                              MPI_Comm c_mpicomm, int argc, char **argv) {
  MPI_Fint f_mpicomm;
  f_mpicomm = MPI_Comm_c2f(c_mpicomm);
  app_main_(smarties_comm, f_mpicomm);
  return 0;
}

int main(int argc, char **argv) {
  smarties::Engine e(argc, argv);
  if (e.parse())
    return 1;
  e.run(app_main_interface);
  return 0;
}
