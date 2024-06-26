#include "smarties.h"
struct Environment {
  void reset(std::mt19937 &){};
  int advance(std::vector<double> action) { return 0; }
  std::vector<double> getState() { return std::vector<double>(); }
  double getReward(void) { return 0.0; };
};

inline void app_main(smarties::Communicator *const comm, int argc,
                     char **argv) {
  int state_dimensionality = 3, action_dimensionality = 2;
  comm->setStateActionDims(state_dimensionality, action_dimensionality);
  Environment env;
  while (true) {
    env.reset(comm->getPRNG());
    comm->sendInitState(env.getState());
    while (true) {
      std::vector<double> action = comm->recvAction();
      bool isTerminal = env.advance(action);
      if (isTerminal) {
        comm->sendTermState(env.getState(), env.getReward());
        break;
      } else
        comm->sendState(env.getState(), env.getReward());
    }
  }
}

int main(int argc, char **argv) {
  smarties::Engine e(argc, argv);
  if (e.parse())
    return 1;
  e.run(app_main);
  return 0;
}
