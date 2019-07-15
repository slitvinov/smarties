#include "Communicators/Communicator.h"
#include <vector>
#include "mpi.h"

//=============================================================================
// Entry point into E.g. fortran code.
extern "C" void extern_app_main(const void* rlcomm, const int f_mpicomm);
//=============================================================================

//=============================================================================
// Program entry point
int app_main(
  Communicator*const rlcomm, // communicator with smarties
  MPI_Comm c_mpicomm,        // mpi_comm that mpi-based apps can use (C handle)
  int argc, char**argv,      // arguments read from app's runtime settings file
) {
  std::cout << "C++ side begins" << std::endl;

  // Convert the C handle to the MPI communicator to a Fortran handle
  MPI_Fint f_mpicomm;
  f_mpicomm = MPI_Comm_c2f(c_mpicomm);

  fortran_app_main(rlcomm, f_mpicomm);

  std::cout << "C++ side ends" << std::endl;
  return 0;
} // main
//=============================================================================

//=============================================================================
extern "C" void smarties_sendInitState(
    void* ptr2comm,
    double *state,
    int array_len
) {
  std::vector<double> state_vec(state, state+array_len);
  static_cast<Communicator*>(ptr2comm)->sendInitState(state_vec);
}

extern "C" void smarties_sendTermState(
    void* ptr2comm,
    double *state,
    int array_len,
    double reward
) {
  std::vector<double> state_vec(state, state+array_len);
  static_cast<Communicator*>(ptr2comm)->sendTermState(state_vec, reward);
}

extern "C" void smarties_sendLastState(
    void* ptr2comm,
    double *state,
    int array_len,
    double reward
) {
  std::vector<double> state_vec(state, state+array_len);
  static_cast<Communicator*>(ptr2comm)->sendLastState(state_vec, reward);
}

extern "C" void smarties_sendState(
    void* ptr2comm,
    double *state,
    int array_len,
    double reward
) {
  std::vector<double> state_vec(state, state+array_len);
  static_cast<Communicator*>(ptr2comm)->sendState(state_vec, reward);
}

extern "C" void smarties_recvAction(
    void* ptr2comm,
    double *action,
    int array_len
) {
  std::vector<double> action_vec(array_len);
  action_vec = static_cast<Communicator*>(ptr2comm)->recvAction();
  // Copy action_vec in action, which is returned to Fortran
  std::copy(action_vec.begin(), action_vec.end(), action);
}
//=============================================================================

//=============================================================================
extern "C" void smarties_set_state_action_dims(
    void* ptr2comm, int state_dim, int action_dim, int agent_id
) {
  static_cast<Communicator*>(ptr2comm)->set_state_action_dims(
    state_dim, action_dim, agent_id);
}

extern "C" void smarties_set_action_scales(
    void * ptr2comm, double * upper_scale, double * lower_scale,
    int are_bounds, int action_dim, int agent_id
) {
  std::vector<double> upper(upper_scale, upper_scale + action_dim);
  std::vector<double> lower(lower_scale, lower_scale + action_dim);
  static_cast<Communicator*>(ptr2comm)->set_action_scales(
    upper, lower, are_bounds, agent_id);
}

//=============================================================================
