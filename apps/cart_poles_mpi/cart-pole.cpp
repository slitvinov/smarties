//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include "smarties.h"
#include "../cart_pole_cpp/cart-pole.h"

#include <iostream>
#include <cstdio>

using namespace std;

int app_main(
  smarties::Communicator*const C, // communicator with smarties
  MPI_Comm mpicom,         // mpi_comm that mpi-based apps can use
  int argc, char**argv    // arguments read from app's runtime settings file
) {
  smarties::Communicator & comm = * C;
  const int control_vars = 1; // force along x
  const int state_vars = 6;
  const int n_agents = 2;
  //  - x position
  //  - x velocity
  //  - ang velocity
  //  - angle
  //  - cos(angle)
  //  - sin(angle)
  comm.set_state_action_dims(state_vars, control_vars);
  comm.set_num_agents(n_agents);

  //OPTIONAL: action bounds
  bool bounded = true;
  vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm.set_action_scales(upper_action_bound, lower_action_bound, bounded);

  /*
    // ALTERNATIVE for discrete actions:
    vector<int> n_options = vector<int>{2};
    comm.set_action_options(n_options);
    // will receive either 0 or 1, app chooses resulting outcome
  */

  //OPTIONAL: hide state variables.
  // e.g. show cosine/sine but not angle
  vector<bool> b_observable = {true, true, true, false, true, true};
  comm.set_state_observable(b_observable);

  //OPTIONAL: set space bounds
  //vector<double> upper_state_bound{ 1,  1,  1,  1,  1,  1};
  //vector<double> lower_state_bound{-1, -1, -1, -1, -1, -1};
  //comm.set_state_scales(upper_state_bound, lower_state_bound);

  comm.agents_define_different_MDP();
  // to make it interesting, one agent is partially observed:
  comm.set_is_partially_observable(1);
  vector<bool> b_observable2 = {true, false, false, false, true, true};
  comm.set_state_observable(b_observable2, 1);

  // Here for simplicity we have two environments
  // But real application is to env with two competing/collaborating agents
  CartPole env1, env2;

  // This function is *needed* to send problem description to smarties ensuring
  // thread safety. if only one thread (e.g. python), then it can be omitted:
  comm.finalize_problem_description();
  omp_set_num_threads(2);

  while(true) //train loop
  {
    //reset environment:
    // getPRNG is not thread safe
    env1.reset(comm.getPRNG()); //comm contains rng with different seed on each rank
    env2.reset(comm.getPRNG()); //comm contains rng with different seed on each rank

    #pragma omp parallel sections
    {
      #pragma omp section
      comm.sendInitState(env1.getState(), 0); //send initial state

      #pragma omp section
      comm.sendInitState(env2.getState(), 1); //send initial state
    }

    while (true) //simulation loop
    {
      std::atomic<bool> terminated1{false}, terminated2{false};
      std::atomic<int> ndone{0};

      #pragma omp parallel sections num_threads(2)
      {
        #pragma omp section
        {
          vector<double> action1 = comm.recvAction(0);
          action1[0] = - action1[0]; // make the two optimal policy different:
          terminated1 = env1.advance(action1);
          vector<double> state1 = env1.getState();
          double reward1 = env1.getReward();
          ++ndone;
          while(ndone.load() < 2) ;
          if(terminated1.load() || terminated2.load()) { // end of simulation
            if(terminated1.load()) comm.sendTermState(state1, reward1, 0);
            else comm.sendLastState(state1, reward1, 0);
          } else comm.sendState(state1, reward1, 0);
        }

        #pragma omp section
        {
          vector<double> action2 = comm.recvAction(1);
          terminated2 = env2.advance(action2);
          vector<double> state2 = env2.getState();
          double reward2 = env2.getReward();
          ++ndone;
          while(ndone.load() < 2) ;
          if(terminated1.load() || terminated2.load()) { // end of simulation
            if(terminated2.load()) comm.sendTermState(state2, reward2, 1);
            else comm.sendLastState(state2, reward2, 1);
          } else comm.sendState(state2, reward2, 1);
        }
      }

      if(terminated1.load() || terminated2.load()) break;
    }
  }
}
