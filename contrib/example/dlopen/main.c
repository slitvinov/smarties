#include <stdint.h>
#include <stdio.h>
#include <mpi.h>
#include <dlfcn.h>
#include <smarties_f77.h>

void *so;
static void*
sym(void)
{
    void *function;
    if ((so = dlopen(NULL, RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND)) == NULL) {
      fprintf(stderr, "%s:%d: dlopen failed: %s\n", __FILE__, __LINE__, dlerror());
      return NULL;
    }
    if ((function = dlsym(so, "app_main_")) == NULL) {
      fprintf(stderr, "%s:%d: dlsym failed: %s\n", __FILE__, __LINE__, dlerror());
      return NULL;
    }
    return function;
}


static int
main0(uintptr_t *smarties, void *mpi0, void *p)
{
    typedef int (*Function)(uintptr_t*, int*, int*);
    MPI_Fint f_mpicomm;
    MPI_Comm *mpi;
    int *i;
    int first;
    FILE *file;
    Function function;

    mpi = (MPI_Comm*)mpi0;
    i = (int*)p;
    fprintf(stderr, "main0: %d\n", *i);
    f_mpicomm = MPI_Comm_c2f(*mpi);

    function = (Function)sym();
    first = 1;
    function(smarties, &f_mpicomm, &first);
    if (dlclose(so) != 0) {
      fprintf(stderr, "%s:%d: dlclose failed: %s\n", __FILE__, __LINE__, dlerror());
      return 2;
    }
    for (;;) {
      function = (Function)sym();
      first = 0;
      function(smarties, &f_mpicomm, &first);
      if (dlclose(so) != 0) {
        fprintf(stderr, "%s:%d: dlclose failed: %s\n", __FILE__, __LINE__, dlerror());
        return 2;
      }
    }
    return 0;
}

int
main(int argc, char **argv)
{
    int i;

    i = 1234;
    return smarties_main_(argc, argv, main0, &i);
}
