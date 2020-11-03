#include <smarties.h>
#include "smarties_f77.h"

extern "C" {
void smarties_sendinitstate_(uintptr_t *i, double *S, int *state_dim,
			     int *agent) {
  void *p;
  p = (void *)(*i);
  std::vector<double> svec(S, S + *state_dim);
  static_cast<smarties::Communicator *>(p)->sendInitState(svec, *agent);
}

void smarties_sendtermstate_(uintptr_t *i, double *S, int *state_dim, double *R,
			     int *agent) {
  void *p;
  p = (void *)(*i);
  std::vector<double> svec(S, S + *state_dim);
  static_cast<smarties::Communicator *>(p)->sendTermState(svec, *R, *agent);
}

void smarties_sendstate_(uintptr_t *i, double *S, int *state_dim, double *R,
			 int *agent) {
  void *p;
  p = (void *)(*i);
  std::vector<double> svec(S, S + *state_dim);
  static_cast<smarties::Communicator *>(p)->sendState(svec, *R, *agent);
}

void smarties_recvaction_(uintptr_t *i, double *A, int *action_dim,
			  int *agent) {
  void *p;
  p = (void *)(*i);
  std::vector<double> avec =
      static_cast<smarties::Communicator *>(p)->recvAction(*agent);
  assert(*action_dim == static_cast<int>(avec.size()));
  std::copy(avec.begin(), avec.end(), A);
}

void smarties_setactionscales_(uintptr_t *i, double *upper_scale,
			       double *lower_scale, int *are_bounds,
			       int *action_dim, int *agent) {
  void *p;
  p = (void *)(*i);
  std::vector<double> upper(upper_scale, upper_scale + *action_dim);
  std::vector<double> lower(lower_scale, lower_scale + *action_dim);
  static_cast<smarties::Communicator *>(p)->setActionScales(
      upper, lower, *are_bounds, *agent);
}

void smarties_setstateobservable_(uintptr_t *i, int *bobservable,
				  int *state_dim, int *agent) {
  void *p;
  p = (void *)(*i);
  std::vector<bool> optionsvec(bobservable, bobservable + *state_dim);
  static_cast<smarties::Communicator *>(p)->setStateObservable(optionsvec,
							       *agent);
}

void smarties_setstatescales_(uintptr_t *i, double *upper_scale,
			      double *lower_scale, int *state_dim, int *agent) {
  void *p;
  p = (void *)(*i);
  std::vector<double> upper(upper_scale, upper_scale + *state_dim);
  std::vector<double> lower(lower_scale, lower_scale + *state_dim);
  static_cast<smarties::Communicator *>(p)->setStateScales(upper, lower,
							   *agent);
}

void smarties_setstateactiondims_(uintptr_t *i, int *state_dim, int *action_dim,
				  int *agent) {
  void *p;
  p = (void *)(*i);
  static_cast<smarties::Communicator *>(p)->setStateActionDims(
      *state_dim, *action_dim, *agent);
}

void smarties_setnumagents_(uintptr_t *i, int *nagent) {
  void *p;
  p = (void *)(*i);
  static_cast<smarties::Communicator *>(p)->setNumAgents(*nagent);
}

int smarties_terminatetraining_(uintptr_t *i) {
  void *p;
  p = (void *)(*i);
  return static_cast<smarties::Communicator *>(p)->terminateTraining();
}

enum {MAGIC = 12345};
struct Data {
  void *data;
  int (*function)(uintptr_t*, void*, void*);
  int magic;
};
static int main0(smarties::Communicator *const smarties_comm,
		 MPI_Comm mpi, int argc, char **argv) {
  int rc;
  struct Data *d;
  uintptr_t i;

  i = (uintptr_t)(smarties_comm);
  if (sscanf(argv[argc - 1], "%p", (void**)&d) != 1) {
    fprintf(stderr, "%s:%d: sscanf cannot get a point\n", __FILE__, __LINE__);
    abort();
  }
  if (d->magic != MAGIC) {
    fprintf(stderr, "%s:%d: wrong pointer to client's data\n", __FILE__, __LINE__);
    abort();
  }
  rc = d->function(&i, &mpi, d->data);
  if (rc != 0) {
    fprintf(stderr, "%s:%d: client failed\n", __FILE__, __LINE__);
    abort();
  }
  return 0;
}

int smarties_main0_(int argc, char **argv0, int nwpe, int (*function)(uintptr_t*, void*, void*), void *data) {
  enum {M_STR = 999, M_ARGC = 999};
  struct Data d;
  char string[M_STR];
  const char *argv[M_ARGC];
  int i;

  if (argc > M_ARGC - 2) {
    fprintf(stderr, "%s:%d: argc=%d is two big\n", __FILE__, __LINE__, argc);
    return 1;
  }
  d.function = function;
  d.data = data;
  d.magic = MAGIC;

  argv[0] = argv0[0];
  for (i = 1; i < argc; i++) {
    if (argv0[i][0] == '-' && argv0[i][1] == '-' && argv0[i][2] == '\0') {
      argv0 += i;
      argc -= i;
      break;
    }
  }
  for (i = 1; i < argc; i++)
    argv[i] = argv0[i];
  if (snprintf(string, M_STR, "%p", (void*)&d) < 0) {
    fprintf(stderr, "%s:%d: snprintf failed\n", __FILE__, __LINE__);
    return 1;
  }
  argv[argc] = string;
  argv[argc + 1] = NULL;
  smarties::Engine e(argc + 1, (char**)argv);
  if (e.parse())
    return 1;
  if (nwpe != 1)
    e.setNworkersPerEnvironment(nwpe);
  e.run(main0);
  return 0;
}

int smarties_main_(int argc, char **argv, int (*function)(uintptr_t*, void*, void*), void *data) {
  int nwpe;
  nwpe = 1;
  return smarties_main0_(argc, argv, nwpe, function, data);
}

}
