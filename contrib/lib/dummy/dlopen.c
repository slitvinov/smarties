#include <dlfcn.h>
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include "smarties_f77.h"

typedef int (*Function)(uintptr_t *, int *, int *);
struct Data {
  const char *filename;
  void *data;
};

static int
dlopen0(uintptr_t *smarties, void *mpi0, void *p)
{
  MPI_Fint f_mpicomm;
  MPI_Comm *mpi;
  int first;
  Function function;
  void *lib;
  struct Data *d;

  mpi = (MPI_Comm *)mpi0;
  d = (struct Data *)p;
  f_mpicomm = MPI_Comm_c2f(*mpi);
  for (first = 1;; first = 0) {
    if ((lib = dlopen(d->filename, RTLD_LAZY | RTLD_LOCAL)) == NULL) {
      fprintf(stderr, "%s:%d: dlopen failed: %s\n", __FILE__, __LINE__,
	      dlerror());
      return 1;
    }
    if ((function = (Function)dlsym(lib, "app_main_")) == NULL) {
      fprintf(stderr, "%s:%d: dlsym failed: %s\n", __FILE__, __LINE__,
	      dlerror());
      return 1;
    }
    if (function(smarties, &f_mpicomm, &first) != 0) {
      fprintf(stderr, "%s:%d: client function failed\n",
	      __FILE__, __LINE__);
      return 1;
    }
    if (dlclose(lib) != 0) {
      fprintf(stderr, "%s:%d: dlclose failed: %s\n", __FILE__, __LINE__,
	      dlerror());
      return 1;
    }
  }
  return 0;
}

int smarties_dlopen_(int argc, char **argv, const char *filename, void *data)
{
  struct Data d;
  d.filename = filename;
  d.data = data;
  return smarties_main_(argc, argv, dlopen0, &d);
}
