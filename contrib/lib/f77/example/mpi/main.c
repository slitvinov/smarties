#include <stdio.h>
#include <stdint.h>
#include <mpi.h>
#include <smarties_f77.h>

static int
main0(uintptr_t *smarties, void *mpi0, void *p)
{
    int flag;
    FILE *file;
    int rank;
    int size;
    MPI_Comm *mpi;

    mpi = (MPI_Comm*)mpi0;
    file = fopen("/dev/tty", "w");
    if (MPI_Initialized(&flag) != 0) {
        fprintf(stderr, "%s:%d: MPI_Initialized failed\n", __FILE__, __LINE__);
        return 1;
    }
    if (flag)
        fprintf(file, "mpi is initialized\n");
    else
        return 1;
    if (MPI_Comm_rank(*mpi, &rank) != 0) {
        fprintf(stderr, "%s:%d: MPI_Comm_rank failed\n", __FILE__, __LINE__);
        return 1;
    }
    if (MPI_Comm_size(*mpi, &size) != 0) {
        fprintf(stderr, "%s:%d: MPI_Comm_size failed\n", __FILE__, __LINE__);
        return 1;
    }
    fprintf(file, "rank/size: %d/%d\n", rank, size);
    return 0;
}

int
main(int argc, const char **argv)
{
    int i;

    i = 42;
    smarties_main_(argc, argv, main0, &i);
}
