#include <dlfcn.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <smarties_f77.h>

static int
main0(uintptr_t *smarties, void *mpi0, void *p)
{
  typedef int (*Function)(uintptr_t *, int *, int *);
  MPI_Fint f_mpicomm;
  MPI_Comm *mpi;
  int first;
  FILE *file;
  Function function;
  void *lib;

  mpi = (MPI_Comm *)mpi0;
  f_mpicomm = MPI_Comm_c2f(*mpi);
  for (first = 1;; first = 0) {
    if ((lib = dlopen("libmain.so", RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND)) == NULL) {
      fprintf(stderr, "%s:%d: dlopen failed: %s\n", __FILE__, __LINE__,
              dlerror());
      return 1;
    }
    if ((function = (Function)dlsym(lib, "app_main_")) == NULL) {
      fprintf(stderr, "%s:%d: dlsym failed: %s\n", __FILE__, __LINE__,
              dlerror());
      return 1;
    }
    function(smarties, &f_mpicomm, &first);
    if (dlclose(lib) != 0) {
      fprintf(stderr, "%s:%d: dlclose failed: %s\n", __FILE__, __LINE__,
              dlerror());
      return 1;
    }
  }
  return 0;
}

int
main(int argc, char **argv)
{
  int i;
  return smarties_main_(argc, argv, main0, &i);
}
