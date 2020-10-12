#include <math.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <mpi.h>
#include <smarties_f77.h>

enum { NCARTS = 2 };
const double mp = 0.1;
const double mc = 1;
const double l = 0.5;
const double g = 9.81;
const double dt = 4e-4;
const int nsteps = 50;
int step = 0;
double F = 0;
double t = 0;
double u[4];
int action_dim = NCARTS;
int state_dim = 4 * NCARTS;
int agent = 0;
uintptr_t smarties;

static double
rnd0(void)
{
  return (double) rand() / (RAND_MAX+1.0);
}

static double
rnd(void)
{
  return 2 * rnd0() - 1;
}

void Diff(double *_u, double *res) {
  double cosy;
  double siny;
  double w;
  double totMass;
  double fac2;
  double F1;
  cosy = cos(_u[2]);
  siny = sin(_u[2]);
  w = _u[3];
  totMass = mp + mc;
  fac2 = l * (4.0 / 3 - (mp * cosy * cosy) / totMass);
  F1 = F + mp * l * w * w * siny;
  res[3] = (g * siny - F1 * cosy / totMass) / fac2;
  res[1] = (F1 - mp * l * res[3] * cosy) / totMass;
  res[0] = _u[1];
  res[2] = _u[3];
}

void rk46_nl(double t0, double dt, double *u0) {
  const double a[] = {0.000000000000,  -0.737101392796, -1.634740794341,
                       -0.744739003780, -1.469897351522, -2.813971388035};
  const double b[] = {0.032918605146, 0.823256998200, 0.381530948900,
                       0.200092213184, 1.718581042715, 0.270000000000};
  const double c[] = {0.000000000000, 0.032918605146, 0.249351723343,
                       0.466911705055, 0.582030414044, 0.847252983783};
  double w[4];
  double res[4];
  double u[4];
  int i;
  int d;
  for (d = 0; d < 4; d++)
    u[d] = u0[d];
  for (i = 0; i < 6; ++i) {
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

static void
reset(void) {
  int d;
  for (d = 0; d < 4; d++)
    u[d] = 0.05*rnd();
  step = 0;
  F = 0;
  t = 0;
}

static int
is_failed(void) {
  return fabs(u[0]) > 2.4 || fabs(u[2]) > M_PI / 15;
}

static int
is_over(void) {
  return step >= 500 || fabs(u[0]) > 2.4 || fabs(u[2]) > M_PI / 15;
}

static int
advance(double action)
{
  F = action;
  step++;
  for (int i = 0; i < nsteps; i++) {
    rk46_nl(t, dt, u);
    t += dt;
    if (is_over())
      return 1;
  }
  return 0;
}

static void
getState(double *state)
{
  int d;
  for (d = 0; d < 4; d++)
    state[d] = u[d];
}

static double
getReward(void) {
  return 1 - (fabs(u[2]) > M_PI / 15 || fabs(u[0]) > 2.4);
}

static int
main0(uintptr_t *smarties0, void *mpi0, void *p)
{
  int d;
  double myState[4];
  int myRank;
  int simSize;
  MPI_Comm mpicom;
  int bounded;
  double upper_action_bound[NCARTS] = {10, 10};
  double lower_action_bound[NCARTS] = {-10, -10};
  double combinedState[4 * NCARTS];
  double combinedAction[action_dim];
  double sumReward;
  double myAction;
  double myReward;
  int nTerminated;
  int myTerminated;

  mpicom = *(MPI_Comm*)mpi0;
  smarties = *smarties0;
  MPI_Comm_rank(mpicom, &myRank);
  MPI_Comm_size(mpicom, &simSize);
  assert(simSize == NCARTS && myRank < NCARTS);
  // This options says that the agent themselves are distributed.
  // I.e. the same agent runs on multiple ranks:
  //comm->envHasDistributedAgents();
  // Because we are holding on to using cart-poles... let's just say that our
  // agent is NCARTS cart-poles with joint controls. 4 state and 1 control
  // variables per process, distributed over NCARTS processes.
  smarties_setstateactiondims_(&smarties, &state_dim, &action_dim, &agent);
  //  comm->setStateActionDims(4 * NCARTS, 1 * NCARTS);

  // OPTIONAL: action bounds
  bounded = 1;
  smarties_setactionscales_(&smarties, upper_action_bound, lower_action_bound, &bounded, &action_dim, &agent);
  MPI_Barrier(mpicom);
  while (1) // train loop
  {
    reset();
    getState(myState);
    for (d = 0; d < 4 * NCARTS; d++)
      combinedState[d] = 0;
    MPI_Allgather(myState, 4, MPI_DOUBLE, combinedState, 4,
		  MPI_DOUBLE, mpicom);
    // Actually, only rank 0 will send the state to smarties.
    // We might as well have used MPI_Gather with root 0.
    smarties_sendinitstate_(&smarties, combinedState, &state_dim, &agent);
    while (1) // simulation loop
    {
      // Each rank will get the same vector here:
      smarties_recvaction_(&smarties, combinedAction, &action_dim, &agent);
      myAction = combinedAction[myRank];
      myTerminated = advance(myAction);
      getState(myState);
      myReward = getReward();

      sumReward = 0;
      nTerminated = 0;

      MPI_Allreduce(&myTerminated, &nTerminated, 1, MPI_INT, MPI_SUM, mpicom);
      MPI_Allreduce(&myReward, &sumReward, 1, MPI_DOUBLE, MPI_SUM, mpicom);
      MPI_Allgather(myState, 4, MPI_DOUBLE, combinedState, 4,
                    MPI_DOUBLE, mpicom);
      if (nTerminated > 0) {
	smarties_sendtermstate_(&smarties, combinedState, &state_dim,
				&sumReward, &agent);
        break;
      } else
        smarties_sendstate_(&smarties, combinedState, &state_dim, &sumReward, &agent);
    }
  }
}

int
main(int argc, char **argv)
{
  return smarties_main_(argc, argv, main0, NULL);
}
