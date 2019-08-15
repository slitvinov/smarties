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

int main()
{
  const int control_vars = 1; // force along x
  const int state_vars = 6;
  const int n_agents = 2;
  //  - x position
  //  - x velocity
  //  - ang velocity
  //  - angle
  //  - cos(angle)
  //  - sin(angle)

  //socket number is given by RL as first argument of execution
  smarties::Communicator comm(state_vars, control_vars, n_agents);

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
  vector<double> upper_state_bound{ 1,  1,  1,  1,  1,  1};
  vector<double> lower_state_bound{-1, -1, -1, -1, -1, -1};
  comm.set_state_scales(upper_state_bound, lower_state_bound);

  comm.agents_define_different_MDP();

  // Here for simplicity we have two environments
  // But real application is to env with two competing/collaborating agents
  CartPole env1, env2;

  while(true) //train loop
  {
    //reset environment:
    env1.reset(comm.getPRNG()); //comm contains rng with different seed on each rank
    env2.reset(comm.getPRNG()); //comm contains rng with different seed on each rank

    comm.sendInitState(env1.getState(), 0); //send initial state
    comm.sendInitState(env2.getState(), 1); //send initial state

    while (true) //simulation loop
    {
      vector<double> action1 = comm.recvAction(0);
      action1[0] = - action1[0]; // make the two optimal policy different
      vector<double> action2 = comm.recvAction(1);

      //advance the simulation:
      bool terminated1 = env1.advance(action1);
      bool terminated2 = env2.advance(action2);

      vector<double> state1 = env1.getState();
      vector<double> state2 = env2.getState();
      double reward1 = env1.getReward();
      double reward2 = env2.getReward();

      if(terminated1 || terminated2)  //tell smarties this is a terminal state
      {
        if(terminated1) comm.sendTermState(state1, reward1, 0);
        else comm.sendLastState(state1, reward1, 0);
        if(terminated2) comm.sendTermState(state2, reward2, 1);
        else comm.sendLastState(state2, reward2, 1);

        break;
      }
      else
      {
        comm.sendState(state1, reward1, 0);
        comm.sendState(state2, reward2, 1);
      }
    }
  }
}
