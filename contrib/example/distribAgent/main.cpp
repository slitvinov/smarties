#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <smarties.h>

enum {NCARTS = 2};
template <typename Func, typename Vec>
Vec rk46_nl(double t0, double dt, Vec u0, Func &&Diff) {
  static double a[] = {0.000000000000,  -0.737101392796,
                                 -1.634740794341, -0.744739003780,
                                 -1.469897351522, -2.813971388035};
  static double b[] = {0.032918605146, 0.823256998200,
                                 0.381530948900, 0.200092213184,
                                 1.718581042715, 0.270000000000};
  static double c[] = {0.000000000000, 0.032918605146,
                       0.249351723343, 0.466911705055,
                       0.582030414044, 0.847252983783};
  static constexpr int s = 6;
  Vec w;
  Vec u(u0);
  double t;

  for (int i = 0; i < s; ++i) {
    t = t0 + dt * c[i];
    w = w * a[i] + Diff(u, t) * dt;
    u = u + w * b[i];
  }
  return u;
}

struct Vec4 {
  double y1, y2, y3, y4;

  Vec4(double _y1 = 0, double _y2 = 0, double _y3 = 0, double _y4 = 0)
      : y1(_y1), y2(_y2), y3(_y3), y4(_y4){};

  Vec4 operator*(double v) const {
    return Vec4(y1 * v, y2 * v, y3 * v, y4 * v);
  }

  Vec4 operator+(const Vec4 &v) const {
    return Vec4(y1 + v.y1, y2 + v.y2, y3 + v.y3, y4 + v.y4);
  }
};

struct CartPole {
  const double mp = 0.1;
  const double mc = 1;
  const double l = 0.5;
  const double g = 9.81;
  const double dt = 4e-4;
  const int nsteps = 50;
  int step = 0;
  Vec4 u;
  double F = 0, t = 0;

  void reset(std::mt19937 &gen) {
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    u = Vec4(dist(gen), dist(gen), dist(gen), dist(gen));
    step = 0;
    F = 0;
    t = 0;
  }

  bool is_failed() {
    return std::fabs(u.y1) > 2.4 || std::fabs(u.y3) > M_PI / 15;
  }
  bool is_over() {
    return step >= 500 || std::fabs(u.y1) > 2.4 || std::fabs(u.y3) > M_PI / 15;
  }

  int advance(std::vector<double> action) {
    F = action[0];
    step++;
    for (int i = 0; i < nsteps; i++) {
      u = rk46_nl(t, dt, u,
                  std::bind(&CartPole::Diff, this, std::placeholders::_1,
                            std::placeholders::_2));
      t += dt;
      if (is_over())
        return 1;
    }
    return 0;
  }

  std::vector<double> getState(const int size = 6) {
    assert(size == 4 || size == 6);
    std::vector<double> state(size);
    state[0] = u.y1;
    state[1] = u.y2;
    state[2] = u.y4;
    state[3] = u.y3;
    if (size == 6) {
      state[4] = std::cos(u.y3);
      state[5] = std::sin(u.y3);
    }
    return state;
  }

  double getReward() {
    return 1 - (std::fabs(u.y3) > M_PI / 15 || std::fabs(u.y1) > 2.4);
  }

  Vec4 Diff(Vec4 _u, double) {
    Vec4 res;

    const double cosy = std::cos(_u.y3), siny = std::sin(_u.y3);
    const double w = _u.y4;
    const double totMass = mp + mc;
    const double fac2 = l * (4.0 / 3 - (mp * cosy * cosy) / totMass);
    const double F1 = F + mp * l * w * w * siny;
    res.y4 = (g * siny - F1 * cosy / totMass) / fac2;
    res.y2 = (F1 - mp * l * res.y4 * cosy) / totMass;
    res.y1 = _u.y2;
    res.y3 = _u.y4;
    return res;
  }
};

static int
app_main(smarties::Communicator *const comm,
         MPI_Comm mpicom,
         int,
         char **
) {
  int myRank, simSize;
  MPI_Comm_rank(mpicom, &myRank);
  MPI_Comm_size(mpicom, &simSize);
  assert(simSize == NCARTS && myRank < NCARTS);
  // This options says that the agent themselves are distributed.
  // I.e. the same agent runs on multiple ranks:
  comm->envHasDistributedAgents();
  // Because we are holding on to using cart-poles... let's just say that our
  // agent is NCARTS cart-poles with joint controls. 4 state and 1 control
  // variables per process, distributed over NCARTS processes.
  comm->setStateActionDims(4 * NCARTS, 1 * NCARTS);

  // OPTIONAL: action bounds
  const bool bounded = true;
  const std::vector<double> upper_action_bound(NCARTS, 10);
  const std::vector<double> lower_action_bound(NCARTS, -10);
  comm->setActionScales(upper_action_bound, lower_action_bound, bounded);

  CartPole env;

  MPI_Barrier(mpicom);
  while (true) // train loop
  {
    {
      // reset environment:
      env.reset(comm->getPRNG());
      const std::vector<double> myState = env.getState(4);
      std::vector<double> combinedState = std::vector<double>(4 * NCARTS);

      MPI_Allgather(myState.data(), 4, MPI_DOUBLE, combinedState.data(), 4,
                    MPI_DOUBLE, mpicom);
      // Actually, only rank 0 will send the state to smarties.
      // We might as well have used MPI_Gather with root 0.
      comm->sendInitState(combinedState);
    }

    while (true) // simulation loop
    {
      // Each rank will get the same vector here:
      const std::vector<double> combinedAction = comm->recvAction();
      assert(combinedAction.size() == NCARTS);
      const std::vector<double> myAction = {combinedAction[myRank]};

      const int myTerminated = env.advance(myAction);
      const std::vector<double> myState = env.getState(4);
      const double myReward = env.getReward();

      std::vector<double> combinedState = std::vector<double>(4 * NCARTS);
      double sumReward = 0;
      int nTerminated = 0;

      MPI_Allreduce(&myTerminated, &nTerminated, 1, MPI_INT, MPI_SUM, mpicom);
      MPI_Allreduce(&myReward, &sumReward, 1, MPI_DOUBLE, MPI_SUM, mpicom);
      MPI_Allgather(myState.data(), 4, MPI_DOUBLE, combinedState.data(), 4,
                    MPI_DOUBLE, mpicom);

      // Environment simulation is distributed across two processes.
      // Still, if one processes says the simulation has terminated
      // it should terminate in all processes! (and then can start anew)
      if (nTerminated > 0) {
        comm->sendTermState(combinedState, sumReward);
        break;
      } else
        comm->sendState(combinedState, sumReward);
    }
  }
}

int main(int argc, char **argv) {
  smarties::Engine e(argc, argv);
  if (e.parse())
    return 1;
  // this app is designed to require NCARTS processes per each env simulation:
  e.setNworkersPerEnvironment(NCARTS);
  e.run(app_main);
  return 0;
}
