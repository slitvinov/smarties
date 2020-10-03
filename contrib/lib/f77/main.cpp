#include "../source/smarties/Communicator.h"

extern "C" {

void smarties_sendinitstate_(void **ptr2comm, double *S, int *state_dim,
                            int *agentID) {
  std::vector<double> svec(S, S + *state_dim);
  static_cast<smarties::Communicator *>(*ptr2comm)->sendInitState(svec, *agentID);
}

void smarties_sendtermstate_(void **ptr2comm, double *S, int *state_dim, double *R,
                            int *agentID) {
  std::vector<double> svec(S, S + *state_dim);
  static_cast<smarties::Communicator *>(*ptr2comm)->sendTermState(svec, *R,
                                                                 *agentID);
}

void smarties_sendstate_(void **ptr2comm, double *S, int *state_dim, double *R,
                        int *agentID) {
  std::vector<double> svec(S, S + *state_dim);
  static_cast<smarties::Communicator *>(*ptr2comm)->sendState(svec, *R, *agentID);
}

void smarties_recvaction_(void **ptr2comm, double *A, int *action_dim,
			   int *agentID) {
  std::vector<double> avec =
      static_cast<smarties::Communicator *>(*ptr2comm)->recvAction(*agentID);
  assert(*action_dim == static_cast<int>(avec.size()));
  std::copy(avec.begin(), avec.end(), A);
}

void smarties_setactionscales_(void **ptr2comm, double *upper_scale,
                              double *lower_scale, int *are_bounds,
                              int *action_dim, int *agent_id) {
  std::vector<double> upper(upper_scale, upper_scale + *action_dim);
  std::vector<double> lower(lower_scale, lower_scale + *action_dim);
  static_cast<smarties::Communicator *>(*ptr2comm)->setActionScales(
      upper, lower, *are_bounds, *agent_id);
}

void smarties_setstateobservable_(void **ptr2comm, int *bobservable,
                                 int *state_dim, int *agent_id) {
  std::vector<bool> optionsvec(bobservable, bobservable + *state_dim);
  static_cast<smarties::Communicator *>(*ptr2comm)->setStateObservable(
      optionsvec, *agent_id);
}

void smarties_setstatescales_(void **ptr2comm, double *upper_scale,
                             double *lower_scale, int *state_dim, int *agent_id) {
  std::vector<double> upper(upper_scale, upper_scale + *state_dim);
  std::vector<double> lower(lower_scale, lower_scale + *state_dim);
  static_cast<smarties::Communicator *>(*ptr2comm)->setStateScales(upper, lower,
                                                                  *agent_id);
}

void smarties_setstateactiondims_(void **ptr2comm, int *state_dim,
                                  int *action_dim, int *agent_id) {
  static_cast<smarties::Communicator *>(*ptr2comm)->setStateActionDims(
      *state_dim, *action_dim, *agent_id);
}  

}
