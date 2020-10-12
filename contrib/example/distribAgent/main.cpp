#include <math.h>
#include <smarties.h>

enum { NCARTS = 2 };
const double mp = 0.1;
const double mc = 1;
const double l = 0.5;
const double g = 9.81;
const double dt = 4e-4;
const int nsteps = 50;
int step = 0;
double F = 0, t = 0;
static double u[4];

void Diff(double *_u, double *res) {
  const double cosy = cos(_u[2]), siny = sin(_u[2]);
  const double w = _u[3];
  const double totMass = mp + mc;
  const double fac2 = l * (4.0 / 3 - (mp * cosy * cosy) / totMass);
  const double F1 = F + mp * l * w * w * siny;
  res[3] = (g * siny - F1 * cosy / totMass) / fac2;
  res[1] = (F1 - mp * l * res[3] * cosy) / totMass;
  res[0] = _u[1];
  res[2] = _u[3];
}

void rk46_nl(double t0, double dt, double *u0) {
  static double a[] = {0.000000000000,  -0.737101392796, -1.634740794341,
                       -0.744739003780, -1.469897351522, -2.813971388035};
  static double b[] = {0.032918605146, 0.823256998200, 0.381530948900,
                       0.200092213184, 1.718581042715, 0.270000000000};
  static double c[] = {0.000000000000, 0.032918605146, 0.249351723343,
                       0.466911705055, 0.582030414044, 0.847252983783};
  int s = 6;
  double w[4];
  double res[4];
  double u[4];
  int i;
  int d;

  for (d = 0; d < 4; d++)
    u[d] = u0[d];
  for (i = 0; i < s; ++i) {
    t = t0 + dt * c[i];
    Diff(u, res);
    for (d = 0; d < 4; d++) {
      w[d] = w[d] * a[i] +  res[d] * dt;
      u[d] = u[d] + w[d] * b[i];
    }
  }
  for (d = 0; d < 4; d++)
    u0[d] = u[d];
}

void reset(std::mt19937 &gen) {
  int d;
  std::uniform_real_distribution<double> dist(-0.05, 0.05);
  for (d = 0; d < 4; d++)
    u[d] = dist(gen);
  step = 0;
  F = 0;
  t = 0;
}

bool is_failed() { return fabs(u[0]) > 2.4 || fabs(u[2]) > M_PI / 15; }
bool is_over() {
  return step >= 500 || fabs(u[0]) > 2.4 || fabs(u[2]) > M_PI / 15;
}

int advance(std::vector<double> action) {
  F = action[0];
  step++;
  for (int i = 0; i < nsteps; i++) {
    rk46_nl(t, dt, u);
    t += dt;
    if (is_over())
      return 1;
  }
  return 0;
}

std::vector<double> getState(const int size = 6) {
  int d;
  assert(size == 4 || size == 6);
  std::vector<double> state(size);
  for (d = 0; d < 4; d++)
    state[d] = u[d];
  if (size == 6) {
    state[4] = cos(u[2]);
    state[5] = sin(u[2]);
  }
  return state;
}

double getReward() { return 1 - (fabs(u[2]) > M_PI / 15 || fabs(u[0]) > 2.4); }

static int app_main(smarties::Communicator *const comm, MPI_Comm mpicom, int,
                    char **) {
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
  MPI_Barrier(mpicom);
  while (true) // train loop
  {
    {
      // reset environment:
      reset(comm->getPRNG());
      const std::vector<double> myState = getState(4);
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

      const int myTerminated = advance(myAction);
      const std::vector<double> myState = getState(4);
      const double myReward = getReward();

      std::vector<double> combinedState = std::vector<double>(4 * NCARTS);
      double sumReward = 0;
      int nTerminated = 0;

      MPI_Allreduce(&myTerminated, &nTerminated, 1, MPI_INT, MPI_SUM, mpicom);
      MPI_Allreduce(&myReward, &sumReward, 1, MPI_DOUBLE, MPI_SUM, mpicom);
      MPI_Allgather(myState.data(), 4, MPI_DOUBLE, combinedState.data(), 4,
                    MPI_DOUBLE, mpicom);
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
  e.setNworkersPerEnvironment(NCARTS);
  e.run(app_main);
  return 0;
}
