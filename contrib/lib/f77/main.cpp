#include "../source/smarties/Communicator.h"
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
}
