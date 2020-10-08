#include <stdio.h>
#include <stdint.h>
#include <smarties_f77.h>

static int
main0(uintptr_t *smarties, void *mpi, void *p)
{
    int *i;
    FILE *file;

    i = p;
    file = fopen("/dev/tty", "w");
    fprintf(file, "magic: %d\n", *i);
    return 0;
}

int
main(int argc, char **argv)
{
    int i;
    i = 42;
    smarties_main_(argc, argv, main0, &i);
}
