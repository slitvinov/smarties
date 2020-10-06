#include <../source/smarties/Communicator.h>
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
  void *v;
  uintptr_t i;
  i = (uintptr_t)(smarties_comm);
  if (sscanf(argv[argc - 1], "%p", &d) != 1) {
    fprintf(stderr, "%s:%d: sscanf cannot get a point\n");
    return 1;
  }
  if (d->magic != MAGIC) {
    fprintf(stderr, "%s:%d: fail to a pointer to client data\n");
    return 1;
  }
  rc = d->function(&i, &mpi, d->data);
  if (rc != 0) {
    fprintf(stderr, "%s:%d: client failed\n", __FILE__, __LINE__);
    return 1;
  }
  return 0;
}

int smarties_main(int argc, char **argv0, int (*function)(uintptr_t*, void*, void*), void *data) {
  enum {SIZE = 9999};
  struct Data d;
  char string[SIZE];
  char *argv[9999];
  int i;
  int rc;

  d.function = function;
  d.data = data;
  d.magic = MAGIC;
  if (argv == NULL) {
    fprintf(stderr, "%s:%d: malloc failed\n", __FILE__, __LINE__);
    return 1;
  }
  for (i = 0; i < argc; i++)
    argv[i] = argv0[i];
  snprintf(string, SIZE, "%p", &d);
  argv[argc] = string;
  smarties::Engine e(argc + 1, argv);
  if (e.parse())
    return 1;
  else
    e.run(main0);
  return 0;
}

}
