#include <cmath>
#include <functional>
#include <random>
#include <smarties.h>
#include <vector>

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

template <typename Func>
Vec4 rk46_nl(double t0, double dt, Vec4 u0, Func &&Diff) {
  static double a[] = {0.000000000000,  -0.737101392796, -1.634740794341,
                       -0.744739003780, -1.469897351522, -2.813971388035};
  static double b[] = {0.032918605146, 0.823256998200, 0.381530948900,
                       0.200092213184, 1.718581042715, 0.270000000000};
  static double c[] = {0.000000000000, 0.032918605146, 0.249351723343,
                       0.466911705055, 0.582030414044, 0.847252983783};
  static int s = 6;
  Vec4 w;
  Vec4 u(u0);
  double t;

  for (int i = 0; i < s; ++i) {
    t = t0 + dt * c[i];
    w = w * a[i] + Diff(u, t) * dt;
    u = u + w * b[i];
  }
  return u;
}

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

  int is_failed() {
    return fabs(u.y1) > 2.4 || fabs(u.y3) > M_PI / 15;
  }
  int is_over() {
    return step >= 500 || fabs(u.y1) > 2.4 || fabs(u.y3) > M_PI / 15;
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
      state[4] = cos(u.y3);
      state[5] = sin(u.y3);
    }
    return state;
  }

  double getReward() {
    return 1 - (fabs(u.y3) > M_PI / 15 || fabs(u.y1) > 2.4);
  }

  Vec4 Diff(Vec4 _u, double _t) {
    Vec4 res;

    const double cosy = cos(_u.y3), siny = sin(_u.y3);
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

inline int
app_main(smarties::Communicator *const comm, // communicator with smarties
         MPI_Comm mpicom, // mpi_comm that mpi-based apps can use
         int argc,
         char **argv // arguments read from app's runtime settings file
) {
  int myRank, simSize;
  MPI_Comm_rank(mpicom, &myRank);
  MPI_Comm_size(mpicom, &simSize);
  const int otherRank = myRank == 0 ? 1 : 0;
  assert(simSize == 2 && myRank < 2); // app designed to be run by 2 ranks

  comm->setStateActionDims(6, 1);

  // OPTIONAL: action bounds
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm->setActionScales(upper_action_bound, lower_action_bound, bounded);
  // OPTIONAL: hide angle, but not cosangle and sinangle.
  std::vector<bool> b_observable = {true, true, true, false, true, true};
  comm->setStateObservable(b_observable);

  CartPole env;

  MPI_Barrier(mpicom);
  while (true) // train loop
  {
    // reset environment:
    env.reset(comm->getPRNG());
    comm->sendInitState(env.getState()); // send initial state

    while (true) // simulation loop
    {
      // advance the simulation:
      const std::vector<double> action = comm->recvAction();

      int terminated[2] = {0, 0};
      terminated[myRank] = env.advance(action);
      MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, terminated, 1, MPI_INT, mpicom);
      const bool myEnvTerminated = terminated[myRank];
      const bool otherTerminated = terminated[otherRank];

      const std::vector<double> state = env.getState();
      const double reward = env.getReward();

      // Environment simulation is distributed across two processes.
      // Still, if one processes says the simulation has terminated
      // it should terminate in all processes! (and then can start anew)
      if (myEnvTerminated || otherTerminated) {
        if (myEnvTerminated)
          comm->sendTermState(state, reward);
        else
          comm->sendLastState(state, reward);
        break;
      } else
        comm->sendState(state, reward);
    }
  }
}

int main(int argc, char **argv) {
  smarties::Engine e(argc, argv);
  if (e.parse())
    return 1;
  // this app is designed to require 2 processes per each env simulation:
  e.setNworkersPerEnvironment(2);
  e.run(app_main);
  return 0;
}
