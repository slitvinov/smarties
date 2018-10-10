//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <functional>
#include "Communicator.h"
using namespace std;

int main(int argc, const char * argv[])
{
  //communication:
  const int socket = std::stoi(argv[1]);

  const int control_vars = 2;
  const int state_vars = 0;

  //socket number is given by RL as first argument of execution
  Communicator comm(socket, state_vars, control_vars);
  auto & G = comm.getPRNG();
  std::uniform_real_distribution<double> dist(-1, 1);
  const double A = 1;
  const double B = 100;
  const auto F = [](const double x, const double y,
                    const double a, const double b) {
     return std::pow(a-x, 2) + b * std::pow(y - x*x, 2);
  }; 
  while (1)
  {
        comm.sendInitState(vector<double>(0));
        vector<double> action = comm.recvAction();
        const double Y = dist(G), X = dist(G);
        const double Z = F(X,Y,A,B);
        const double z = F(X,Y,action[0],action[1]);
        const double R = - std::pow(Z-z, 2);
        comm.sendTermState(vector<double>(0), R);
  }
  return 0;
}
