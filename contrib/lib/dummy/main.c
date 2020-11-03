#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <mpi.h>
#include <smarties_f77.h>

void smarties_sendinitstate_(uintptr_t *i, double *S, int *state_dim,
			     int *agent) {}

void smarties_sendtermstate_(uintptr_t *i, double *S, int *state_dim, double *R,
			     int *agent) {}

void smarties_sendstate_(uintptr_t *i, double *S, int *state_dim, double *R,
			 int *agent) {}

void smarties_recvaction_(uintptr_t *i, double *A, int *action_dim,
			  int *agent) {}

void smarties_setactionscales_(uintptr_t *i, double *upper_scale,
			       double *lower_scale, int *are_bounds,
			       int *action_dim, int *agent) {}

void smarties_setstateobservable_(uintptr_t *i, int *bobservable,
				  int *state_dim, int *agent) {}

void smarties_setstatescales_(uintptr_t *i, double *upper_scale,
			      double *lower_scale, int *state_dim, int *agent) {
}

void smarties_setstateactiondims_(uintptr_t *i, int *state_dim, int *action_dim,
				  int *agent) {}

void smarties_setnumagents_(uintptr_t *i, int *nagent) {
}

int smarties_terminatetraining_(uintptr_t *i) { return 0; }

int smarties_main0_(int argc, char **argv, int nwpe,
		    int (*function)(uintptr_t *, void *, void *), void *data) {
  int rc;
  int prov;
  char string[MPI_MAX_ERROR_STRING];
  uintptr_t smarties;
  int resultlen;
  int size;
  MPI_Comm mpi;

  mpi = MPI_COMM_WORLD;
  if ((rc = MPI_Init_thread(&argc, (char ***)&argv, MPI_THREAD_MULTIPLE,
			    &prov)) != MPI_SUCCESS) {
    MPI_Error_string(rc, string, &resultlen);
    fprintf(stderr, "%s:%d: mpi failed: %s\n", __FILE__, __LINE__, string);
    abort();
  }
  if ((rc = MPI_Comm_size(mpi, &size)) != MPI_SUCCESS) {
    MPI_Error_string(rc, string, &resultlen);
    fprintf(stderr, "%s:%d: mpi failed: %s\n", __FILE__, __LINE__, string);
    abort();
  }
  if (size != nwpe) {
    fprintf(stderr, "%s:%d: size=%d != nwpe=%d\n", __FILE__, __LINE__, size, nwpe);
    abort();
  }
  if (function(&smarties, &mpi, data) != 0) {
    fprintf(stderr, "%s:%d: (function) failed\n", __FILE__, __LINE__);
    abort();
  }
  if (MPI_Finalize() != MPI_SUCCESS) {
    MPI_Error_string(rc, string, &resultlen);
    fprintf(stderr, "%s:%d: mpi failed: %s\n", __FILE__, __LINE__, string);
    abort();
  }
  return 0;
}

int smarties_main_(int argc, char **argv,
		   int (*function)(uintptr_t *, void *, void *), void *data) {
  int nwpe;
  nwpe = 1;
  return smarties_main0_(argc, argv, nwpe, function, data);
}
