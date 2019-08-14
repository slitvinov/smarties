//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Communicator_h
#define smarties_Communicator_h

#include "../Core/Environment.h"
#include <random>

namespace smarties
{
struct COMM_buffer;
class Communicator;
class Worker;
}

// main function callback to user's application
// arguments are: - the communicator with smarties
//                - (optional) the mpi communicator to use within the app
//                - argc and argv read from settings file

using environment_callback_t =
  std::function<void(
      smarties::Communicator*const smartiesCommunicator,
      int argc, char**argv
    )>;

using environment_callback_MPI_t =
std::function<void(
    smarties::Communicator*const smartiesCommunicator,
    const MPI_Comm mpiCommunicator,
    int argc, char**argv
  )>;

namespace smarties
{
class Communicator
{
public:
  //Communicator(int number_of_agents = 1);
  //Communicator(int stateDim, int actionDim, int number_of_agents = 1);

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// BEGINNER METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  // Send first state of an episode in order to get first action:
  __attribute__((visibility("default")))
  void sendInitState(const std::vector<double>& state,
                     const int agentID=0)
  {
    return _sendState(agentID, INIT, state, 0);
  }

  // Send normal state and reward:
  __attribute__((visibility("default")))
  void sendState(const std::vector<double>& state,
                 const double reward,
                 const int agentID = 0)
  {
    return _sendState(agentID, CONT, state, reward);
  }

  // Send terminal state/reward: the last step of an episode which ends because
  // of TERMINATION (e.g. agent cannot continue due to failure or success).
  __attribute__((visibility("default")))
  void sendTermState(const std::vector<double>& state,
                    const double reward,
                    const int agentID = 0)
  {
    return _sendState(agentID, TERM, state, reward);
  }

  // Send truncated state/reward: the last step of an episode which ends because
  // of TRUNCATION (e.g. agent cannot continue due to time limits). Difference
  // from TERMINATION is that policy was not direct cause of episode's end.
  __attribute__((visibility("default")))
  void sendLastState(const std::vector<double>& state,
                     const double reward,
                     const int agentID = 0)
  {
    return _sendState(agentID, TRNC, state, reward);
  }

  // receive action for the latest given state:
  __attribute__((visibility("default")))
  const std::vector<double>& recvAction(const int agentID = 0) const;

  __attribute__((visibility("default")))
  void set_state_action_dims(const int dimState, const int dimAct,
                             const int agentID = 0);

  __attribute__((visibility("default")))
  void set_action_scales(const std::vector<double> uppr,
                         const std::vector<double> lowr,
                         const bool bound,
                         const int agentID = 0);

  __attribute__((visibility("default")))
  void set_action_scales(const std::vector<double> upper,
                         const std::vector<double> lower,
                         const std::vector<bool>   bound,
                         const int agentID = 0);

  __attribute__((visibility("default")))
  void set_action_options(const int options,
                          const int agentID = 0);

  __attribute__((visibility("default")))
  void set_action_options(const std::vector<int> options,
                          const int agentID = 0);

  __attribute__((visibility("default")))
  void set_state_observable(const std::vector<bool> observable,
                            const int agentID = 0);

  __attribute__((visibility("default")))
  void set_state_scales(const std::vector<double> upper,
                        const std::vector<double> lower,
                        const int agentID = 0);

  __attribute__((visibility("default")))
  void set_num_agents(int _nAgents);

  __attribute__((visibility("default")))
  void set_is_partially_observable(const int agentID = 0);

  __attribute__((visibility("default")))
  void finalize_problem_description();

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// ADVANCED METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

  __attribute__((visibility("default")))
  void env_has_distributed_agents();

  __attribute__((visibility("default")))
  void agents_define_different_MDP();

  __attribute__((visibility("default")))
  void disableDataTrackingForAgents(int agentStart, int agentEnd);

  __attribute__((visibility("default")))
  void set_preprocessing_conv2d(
    const int input_width, const int input_height, const int input_features,
    const int kernels_num, const int filters_size, const int stride,
    const int agentID = 0);

  __attribute__((visibility("default")))
  void set_num_appended_past_observations(const int n_appended,
                                          const int agentID = 0);

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////// OPTIMIZATION INTERFACE ////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  // conveniency methods for optimization (stateless/timeless) problems

  __attribute__((visibility("default")))
  const std::vector<double>& getOptimizationParameters(const int agentID = 0)
  {
    assert(ENV.descriptors[agentID]->dimState == 0 &&
           "optimization interface only defined for stateless problems");
    _sendState(agentID, INIT, std::vector<double>(0), 0); // fake initial state
    return recvAction(agentID);
  }

  __attribute__((visibility("default")))
  void setOptimizationEvaluation(const Real R, const int agentID = 0)
  {
    assert(ENV.descriptors[agentID]->dimState == 0 &&
           "optimization interface only defined for stateless problems");
    _sendState(agentID, TERM, std::vector<double>(0), R); // send objective eval
  }

  //////////////////////////////////////////////////////////////////////////////
  ////////////////////////////// UTILITY METHODS ///////////////////////////////
  //////////////////////////////////////////////////////////////////////////////
  __attribute__((visibility("default")))
  std::mt19937& getPRNG();

  __attribute__((visibility("default")))
  bool isTraining() const;

  __attribute__((visibility("default")))
  bool terminateTraining() const;

  //////////////////////////////////////////////////////////////////////////////
  ///////////////////////////// DEVELOPER METHODS //////////////////////////////
  //////////////////////////////////////////////////////////////////////////////

protected:
  bool bEnvDistributedAgents = false;

  Environment ENV;
  std::vector<std::unique_ptr<Agent>>& agents = ENV.agents;
  std::vector<std::unique_ptr<COMM_buffer>> BUFF;

  struct {
    int server;
    std::vector<int> clients;
  } SOCK;

  void synchronizeEnvironments();
  void initOneCommunicationBuffer();
  //random number generation:
  std::mt19937 gen;
  //internal counters & flags
  bool bTrain = true;
  bool bTrainIsOver = false;
  long nRequestedEnvTimeSteps = -1;
  Uint globalTstepCounter = 0;

  //called by app to interact with smarties:
  __attribute__((visibility("default")))
  void _sendState(const int agentID, const episodeStatus status,
    const std::vector<double>& state, const double reward);

  //access to smarties' internals, available only if app is linked into exec
  friend class Worker;

  Worker * const worker = nullptr;

  Communicator(Worker* const, std::mt19937&, bool);
};

struct COMM_buffer
{
  COMM_buffer(const size_t maxSdim, const size_t maxAdim) :
    maxStateDim(maxSdim), maxActionDim(maxAdim),
    sizeStateMsg(Agent::computeStateMsgSize(maxSdim)),
    sizeActionMsg(Agent::computeActionMsgSize(maxAdim)),
    dataStateBuf (malloc(sizeStateMsg) ), // aligned_alloc(1024...)
    dataActionBuf(malloc(sizeActionMsg)) { }

  ~COMM_buffer() {
    assert(dataStateBuf not_eq nullptr && dataActionBuf not_eq nullptr);
    free(dataActionBuf);
    free(dataStateBuf);
  }

  COMM_buffer(const COMM_buffer& c) = delete;
  COMM_buffer& operator= (const COMM_buffer& s) = delete;

  const size_t maxStateDim, maxActionDim, sizeStateMsg, sizeActionMsg;
  void * const dataStateBuf;
  void * const dataActionBuf;
};

} // end namespace smarties
#endif // smarties_Communicator_h
