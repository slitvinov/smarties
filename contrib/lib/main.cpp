#include "../source/smarties/Communicator.h"

extern "C" {

void smarties_sendInitState(void *ptr2comm, double *S, int state_dim,
                            int agentID) {
  std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->sendInitState(svec, agentID);
}

void smarties_sendTermState(void *ptr2comm, double *S, int state_dim, double R,
                            int agentID) {
  std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->sendTermState(svec, R,
                                                                 agentID);
}

void smarties_sendLastState(void *ptr2comm, double *S, int state_dim, double R,
                            int agentID) {
  std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->sendLastState(svec, R,
                                                                 agentID);
}

void smarties_sendState(void *ptr2comm, double *S, int state_dim, double R,
                        int agentID) {
  std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->sendState(svec, R, agentID);
}

void smarties_recvAction(void *ptr2comm, double *A, int action_dim,
                         int agentID) {
  std::vector<double> avec =
      static_cast<smarties::Communicator *>(ptr2comm)->recvAction(agentID);
  assert(action_dim == static_cast<int>(avec.size()));
  std::copy(avec.begin(), avec.end(), A);
}
void smarties_setNumAgents(void *ptr2comm, int num_agents) {
  static_cast<smarties::Communicator *>(ptr2comm)->setNumAgents(num_agents);
}

void smarties_setStateActionDims(void *ptr2comm, int state_dim, int action_dim,
                                 int agent_id) {
  FILE *f;
  f = fopen("log", "w");
  fprintf(f, "agent_id: %d %d %d\n", state_dim, action_dim, agent_id);
  static_cast<smarties::Communicator *>(ptr2comm)->setStateActionDims(
      state_dim, action_dim, agent_id);
  fclose(f);
}

void smarties_setActionScales(void *ptr2comm, double *upper_scale,
                              double *lower_scale, int are_bounds,
                              int action_dim, int agent_id) {
  std::vector<double> upper(upper_scale, upper_scale + action_dim);
  std::vector<double> lower(lower_scale, lower_scale + action_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setActionScales(
      upper, lower, are_bounds, agent_id);
}

void smarties_setActionScalesBounds(void *ptr2comm, double *upper_scale,
                                    double *lower_scale, int *are_bounds,
                                    int action_dim, int agent_id) {
  std::vector<double> upper(upper_scale, upper_scale + action_dim);
  std::vector<double> lower(lower_scale, lower_scale + action_dim);
  std::vector<bool> bounds(are_bounds, are_bounds + action_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setActionScales(
      upper, lower, bounds, agent_id);
}

void smarties_setActionOptions(void *ptr2comm, int noptions, int agent_id) {
  static_cast<smarties::Communicator *>(ptr2comm)->setActionOptions(noptions,
                                                                    agent_id);
}

void smarties_setActionOptionsPerDim(void *ptr2comm, int *noptions,
                                     int action_dim, int agent_id) {
  std::vector<int> optionsvec(noptions, noptions + action_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setActionOptions(optionsvec,
                                                                    agent_id);
}

void smarties_setStateObservable(void *ptr2comm, int *bobservable,
                                 int state_dim, int agent_id) {
  std::vector<bool> optionsvec(bobservable, bobservable + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setStateObservable(
      optionsvec, agent_id);
}

void smarties_setStateScales(void *ptr2comm, double *upper_scale,
                             double *lower_scale, int state_dim, int agent_id) {
  std::vector<double> upper(upper_scale, upper_scale + state_dim);
  std::vector<double> lower(lower_scale, lower_scale + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setStateScales(upper, lower,
                                                                  agent_id);
}

void smarties_setIsPartiallyObservable(void *ptr2comm, int agent_id) {
  static_cast<smarties::Communicator *>(ptr2comm)->setIsPartiallyObservable(
      agent_id);
}

void smarties_finalizeProblemDescription(void *ptr2comm) {
  static_cast<smarties::Communicator *>(ptr2comm)->finalizeProblemDescription();
}

void smarties_envHasDistributedAgents(void *ptr2comm) {
  static_cast<smarties::Communicator *>(ptr2comm)->envHasDistributedAgents();
}

void smarties_agentsDefineDifferentMDP(void *ptr2comm) {
  static_cast<smarties::Communicator *>(ptr2comm)->agentsDefineDifferentMDP();
}

void smarties_disableDataTrackingForAgents(void *ptr2comm, int agentStart,
                                           int agentEnd) {
  static_cast<smarties::Communicator *>(ptr2comm)->disableDataTrackingForAgents(
      agentStart, agentEnd);
}

void smarties_setPreprocessingConv2d(void *ptr2comm, int input_width,
                                     int input_height, int input_features,
                                     int kernels_num, int filters_size,
                                     int stride, int agentID) {
  static_cast<smarties::Communicator *>(ptr2comm)->setPreprocessingConv2d(
      input_width, input_height, input_features, kernels_num, filters_size,
      stride, agentID);
}

void smarties_setNumAppendedPastObservations(void *ptr2comm, int n_appended,
                                             int agentID) {
  static_cast<smarties::Communicator *>(ptr2comm)
      ->setNumAppendedPastObservations(n_appended, agentID);
}

void smarties_getUniformRandom(void *ptr2comm, double begin, double end,
                               double *sampled) {
  (*sampled) =
      static_cast<smarties::Communicator *>(ptr2comm)->getUniformRandom(begin,
                                                                        end);
}

void smarties_getNormalRandom(void *ptr2comm, double mean, double stdev,
                              double *sampled) {
  (*sampled) = static_cast<smarties::Communicator *>(ptr2comm)->getNormalRandom(
      mean, stdev);
}

void smarties2_setstateactiondims_(void **ptr2comm, int *state_dim,
                                  int *action_dim, int *agent_id) {
  static_cast<smarties::Communicator *>(*ptr2comm)->setStateActionDims(
      *state_dim, *action_dim, *agent_id);
}  

}
