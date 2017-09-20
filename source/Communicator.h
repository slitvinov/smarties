#pragma once

#include <sstream>
#include <sys/un.h>
#ifdef __RL_MPI_CLIENT
#include <mpi.h>
#endif
#ifdef __Smarties_
#include "Settings.h"
#endif
#include <vector>
#include <cstring>
#include <random>
#define envInfo  int
#define CONT_COMM 0
#define INIT_COMM 1
#define TERM_COMM 2
#define GAME_OVER 3
#define FAIL_COMM 4
#define _AGENT_KILLSIGNAL -256
#ifdef OPEN_MPI
#define MPI_INCLUDED
#endif
#include "Communicator_utils.h"


class Communicator
{
protected:
  //App output file descriptor
  int fd;
  fpos_t pos;

  //Communication over sockets
  int socket_id, Socket, ServerSocket;
  char SOCK_PATH[256];
  struct sockaddr_un serverAddress, clientAddress;

  void print()
  {
    std::ostringstream o;
    o <<(spawner?"Server":"Client")<<" communicator on ";
    o <<(called_by_app?"app":"smarties")<<" side:\n";
    o <<"nStates:"<<nStates<<" nActions:"<<nActions;
    o <<" size_action:"<<size_action<< " size_state:"<< size_state<<"\n";
    o <<"MPI comm: size_s"<<size_learn_pool<<" rank_s:"<<rank_learn_pool;
    o <<" size_a:"<<size_inside_app<< " rank_a:"<< rank_inside_app<<"\n";
    o <<"Socket comm: prefix:"<<socket_id<<" PATH:"<<SOCK_PATH<<"\n";
    std::cout<<o.str()<<std::endl;
  }

  void update_rank_size()
  {
    #ifdef MPI_INCLUDED
    if (comm_inside_app != MPI_COMM_NULL) {
      MPI_Comm_rank(comm_inside_app, &rank_inside_app);
      MPI_Comm_size(comm_inside_app, &size_inside_app);
    }
    if (comm_learn_pool != MPI_COMM_NULL) {
      MPI_Comm_rank(comm_learn_pool, &rank_learn_pool);
      MPI_Comm_size(comm_learn_pool, &size_learn_pool);
    }
    #endif
  }

  void printLog(const double*const buf, const int size);
  void printBuf(const double*const ptr, const int size);

  void launch_exec(const std::string exec);
  void launch_smarties();
  void launch_app();
  void setupClient();
  void setupServer();
  void redirect_stdout_stderr();

public:
  #ifdef MPI_INCLUDED
    MPI_Comm comm_inside_app = MPI_COMM_NULL, comm_learn_pool = MPI_COMM_NULL;
  #endif
  int rank_inside_app = -1, rank_learn_pool = -1;
  int size_inside_app = -1, size_learn_pool = -1;
  int slaveGroup = -1;
  int nStates = -1, nActions = -1, size_state = -1, size_action = -1;

  // communication buffers:
  double *data_state = nullptr, *data_action = nullptr;
  bool called_by_app = false, defined_spaces = false;

  //internal counters
  unsigned long seq_id = 0, msg_id = 0, iter = 0, lag = 1;
  bool verbose = false, spawner = false;
  std::string execpath = std::string();
  std::string paramfile = std::string();
  std::string logfile = std::string();
  std::mt19937 gen;

  int discrete_action_values = 0;
  double dump_value = -1;
  bool sentStateActionShape = false;
  std::vector<double> obs_bounds, obs_inuse, action_options, action_bounds;

  void update_state_action_dims(const int sdim, const int adim);
  void set_logfile(const std::string fname, const bool print_ascii = false)
  {
    logfile = fname;
    verbose = print_ascii;
  }
  void set_action_scales(const std::vector<double> upper,
    const std::vector<double> lower, const bool bound = false);
  void set_action_options(const std::vector<int> options);
  void set_state_scales(const std::vector<double> upper,
    const std::vector<double> lower);
  void set_state_observable(const std::vector<bool> observable);
  void getStateActionShape();
  void sendStateActionShape();

  int getStateDim()  {return nStates;}
  int getActionDim() {return nActions;}
  //called by app to interact with smarties
  void sendCompleteTermination();
  void sendState(const int iAgent, const envInfo status,
    const std::vector<double> state, const double reward);
  //simplified functions:
  inline void sendState(const std::vector<double> state, const double reward, const int iAgent = 0)
  {
    return sendState(iAgent, CONT_COMM, state, reward);
  }
  inline void sendInitState(const std::vector<double> state, const int iAgent = 0)
  {
    return sendState(iAgent, INIT_COMM, state, 0);
  }
  inline void sendTermState(const std::vector<double> state, const double reward, const int iAgent = 0)
  {
    return sendState(iAgent, TERM_COMM, state, reward);
  }
  void recvAction(std::vector<double>& actions);

  std::vector<double> recvAction()
  {
    assert(nActions>0);
    std::vector<double> act(nActions);
    recvAction(act);
    return act;
  }

  void launch();

  Communicator(const int socket, const int sdim, const int adim);

#ifdef MPI_INCLUDED
  Communicator(const int socket, const int sdim, const int adim, const MPI_Comm app);
#endif

  ~Communicator();

#ifdef __Smarties_
  int recvStateFromApp();
  int sendActionToApp();

  double* getDataAction() { return data_action; }
  double* getDataState()  { return data_state; }
  void answerTerminateReq(const double answer);
  void set_params_file(const std::string fname) { paramfile = fname; }
  void set_exec_path(const std::string fname) { execpath = fname; }
  void set_application_mpicom(const MPI_Comm acom, const int group) {
    comm_inside_app = acom;
    slaveGroup = group;
    update_rank_size();
  }

  void restart(std::string fname);
  void save() const;

  void ext_app_run();
  int jobs_init(char *line, char **largv);
  void redirect_stdout_init();
  void redirect_stdout_finalize();
  //called by smarties
  Communicator(const MPI_Comm scom, const int socket, const bool spawn);

  //tmp:
  void send_buffer_to_app(double*const buf, const int size) const
  {
    comm_sock(Socket, true, buf, size);
  }
  void recv_buffer_to_app(double*const buf, const int size) const
  {
    comm_sock(Socket, false, buf, size);
  }
#endif
};

inline void unpackState(double* const data, int& agent, envInfo& info,
    std::vector<double>& state, double& reward)
{
  assert(data not_eq nullptr);
  agent = doublePtrToInt(data+0);
  info  = doublePtrToInt(data+1);
  for (unsigned j=0; j<state.size(); j++) {
    state[j] = data[j+2];
    assert(not std::isnan(state[j]));
    assert(not std::isinf(state[j]));
  }
  reward = data[state.size()+2];
  assert(not std::isnan(reward));
  assert(not std::isinf(reward));
}
