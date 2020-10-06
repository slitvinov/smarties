#include <stdint.h>
#include <stdio.h>
#include <mpi.h>
#include <smarties_f77.h>

extern int app_main_(uintptr_t*, int *);
static int
main0(uintptr_t *smarties, void *mpi0, void *p)
{
    MPI_Fint f_mpicomm;
    MPI_Comm *mpi;
    int *i;
    FILE *file;

    mpi = (MPI_Comm*)mpi0;
    i = (int*)p;
    fprintf(stderr, "main0: %d\n", *i);
    f_mpicomm = MPI_Comm_c2f(*mpi);
    return app_main_(smarties, &f_mpicomm);
}

int main(int argc, char **argv)
{
    int i;

    i = 1234;
    return smarties_main_(argc, argv, main0, &i);
}
