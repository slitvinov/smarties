#include <stdint.h>
#include <stdio.h>
#include <smarties_f77.h>

void smarties_sendinitstate_(uintptr_t *i, double *S, int *state_dim,
                             int *agent) {
}

void smarties_sendtermstate_(uintptr_t *i, double *S, int *state_dim, double *R,
                             int *agent) {
}

void smarties_sendstate_(uintptr_t *i, double *S, int *state_dim, double *R,
                         int *agent) {
}

void smarties_recvaction_(uintptr_t *i, double *A, int *action_dim,
                          int *agent) {
}

void smarties_setactionscales_(uintptr_t *i, double *upper_scale,
                               double *lower_scale, int *are_bounds,
                               int *action_dim, int *agent) {
}

void smarties_setstateobservable_(uintptr_t *i, int *bobservable,
                                  int *state_dim, int *agent) {
}

void smarties_setstatescales_(uintptr_t *i, double *upper_scale,
                              double *lower_scale, int *state_dim, int *agent) {
}

void smarties_setstateactiondims_(uintptr_t *i, int *state_dim, int *action_dim,
                                  int *agent) {
}

int smarties_main_(int argc, char **argv0, int (*function)(uintptr_t*, void*, void*), void *data) {
    int i;
    uintptr_t smarties;
    int mpi;
    if (function(&smarties, &mpi, data) != 0) {
        fprintf(stderr, "%s:%d: (function) failed\n", __FILE__, __LINE__);
        return 1;
    }
    return 0;
}
