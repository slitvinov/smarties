#include <smarties_extern.h>
extern "C" {

void smarties_sendInitState(void *const ptr2comm, const double *const S,
                            const int state_dim, const int agentID) {
  const std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->sendInitState(svec, agentID);
}

void smarties_sendTermState(void *const ptr2comm, const double *const S,
                            const int state_dim, const double R,
                            const int agentID) {
  const std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->sendTermState(svec, R,
                                                                 agentID);
}

void smarties_sendLastState(void *const ptr2comm, const double *const S,
                            const int state_dim, const double R,
                            const int agentID) {
  const std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->sendLastState(svec, R,
                                                                 agentID);
}

void smarties_sendState(void *const ptr2comm, const double *const S,
                        const int state_dim, const double R,
                        const int agentID) {
  const std::vector<double> svec(S, S + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->sendState(svec, R, agentID);
}

void smarties_recvAction(void *const ptr2comm, double *const A,
                         const int action_dim, const int agentID) {
  const std::vector<double> avec =
      static_cast<smarties::Communicator *>(ptr2comm)->recvAction(agentID);
  assert(action_dim == static_cast<int>(avec.size()));
  std::copy(avec.begin(), avec.end(), A);
}
void smarties_setNumAgents(void *const ptr2comm, const int num_agents) {
  static_cast<smarties::Communicator *>(ptr2comm)->setNumAgents(num_agents);
}

void smarties_setStateActionDims(void *const ptr2comm, const int state_dim,
                                 const int action_dim, const int agent_id) {
  static_cast<smarties::Communicator *>(ptr2comm)->setStateActionDims(
      state_dim, action_dim, agent_id);
}

void smarties_setActionScales(void *const ptr2comm,
                              const double *const upper_scale,
                              const double *const lower_scale,
                              const int are_bounds, const int action_dim,
                              const int agent_id) {
  const std::vector<double> upper(upper_scale, upper_scale + action_dim);
  const std::vector<double> lower(lower_scale, lower_scale + action_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setActionScales(
      upper, lower, are_bounds, agent_id);
}

void smarties_setActionScalesBounds(void *const ptr2comm,
                                    const double *const upper_scale,
                                    const double *const lower_scale,
                                    const int *const are_bounds,
                                    const int action_dim, const int agent_id) {
  const std::vector<double> upper(upper_scale, upper_scale + action_dim);
  const std::vector<double> lower(lower_scale, lower_scale + action_dim);
  const std::vector<bool> bounds(are_bounds, are_bounds + action_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setActionScales(
      upper, lower, bounds, agent_id);
}

void smarties_setActionOptions(void *const ptr2comm, const int noptions,
                               const int agent_id) {
  static_cast<smarties::Communicator *>(ptr2comm)->setActionOptions(noptions,
                                                                    agent_id);
}

void smarties_setActionOptionsPerDim(void *const ptr2comm,
                                     const int *const noptions,
                                     const int action_dim, const int agent_id) {
  const std::vector<int> optionsvec(noptions, noptions + action_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setActionOptions(optionsvec,
                                                                    agent_id);
}

void smarties_setStateObservable(void *const ptr2comm,
                                 const int *const bobservable,
                                 const int state_dim, const int agent_id) {
  const std::vector<bool> optionsvec(bobservable, bobservable + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setStateObservable(
      optionsvec, agent_id);
}

void smarties_setStateScales(void *const ptr2comm,
                             const double *const upper_scale,
                             const double *const lower_scale,
                             const int state_dim, const int agent_id) {
  const std::vector<double> upper(upper_scale, upper_scale + state_dim);
  const std::vector<double> lower(lower_scale, lower_scale + state_dim);
  static_cast<smarties::Communicator *>(ptr2comm)->setStateScales(upper, lower,
                                                                  agent_id);
}

void smarties_setIsPartiallyObservable(void *const ptr2comm,
                                       const int agent_id) {
  static_cast<smarties::Communicator *>(ptr2comm)->setIsPartiallyObservable(
      agent_id);
}

void smarties_finalizeProblemDescription(void *const ptr2comm) {
  static_cast<smarties::Communicator *>(ptr2comm)->finalizeProblemDescription();
}

void smarties_envHasDistributedAgents(void *const ptr2comm) {
  static_cast<smarties::Communicator *>(ptr2comm)->envHasDistributedAgents();
}

void smarties_agentsDefineDifferentMDP(void *const ptr2comm) {
  static_cast<smarties::Communicator *>(ptr2comm)->agentsDefineDifferentMDP();
}

void smarties_disableDataTrackingForAgents(void *const ptr2comm,
                                           const int agentStart,
                                           const int agentEnd) {
  static_cast<smarties::Communicator *>(ptr2comm)->disableDataTrackingForAgents(
      agentStart, agentEnd);
}

void smarties_setPreprocessingConv2d(
    void *const ptr2comm, const int input_width, const int input_height,
    const int input_features, const int kernels_num, const int filters_size,
    const int stride, const int agentID) {
  static_cast<smarties::Communicator *>(ptr2comm)->setPreprocessingConv2d(
      input_width, input_height, input_features, kernels_num, filters_size,
      stride, agentID);
}

void smarties_setNumAppendedPastObservations(void *const ptr2comm,
                                             const int n_appended,
                                             const int agentID) {
  static_cast<smarties::Communicator *>(ptr2comm)
      ->setNumAppendedPastObservations(n_appended, agentID);
}

void smarties_getUniformRandom(void *const ptr2comm, const double begin,
                               const double end, double *sampled) {
  (*sampled) =
      static_cast<smarties::Communicator *>(ptr2comm)->getUniformRandom(begin,
                                                                        end);
}

void smarties_getNormalRandom(void *const ptr2comm, const double mean,
                              const double stdev, double *sampled) {
  (*sampled) = static_cast<smarties::Communicator *>(ptr2comm)->getNormalRandom(
      mean, stdev);
}
}
