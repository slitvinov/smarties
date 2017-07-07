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
#define SWINGUP 1
using namespace std;

// Julien Berland, Christophe Bogey, Christophe Bailly,
// Low-dissipation and low-dispersion fourth-order Runge-Kutta algorithm,
// Computers & Fluids, Volume 35, Issue 10, December 2006, Pages 1459-1463, ISSN 0045-7930,
// http://dx.doi.org/10.1016/j.compfluid.2005.04.003
template <typename Func, typename Vec>
Vec rk46_nl(double t0, double dt, Vec u0, Func&& Diff)
{
    const double a[] = {0.000000000000, -0.737101392796, -1.634740794341, -0.744739003780, -1.469897351522, -2.813971388035};
    const double b[] = {0.032918605146,  0.823256998200,  0.381530948900,  0.200092213184,  1.718581042715,  0.270000000000};
    const double c[] = {0.000000000000,  0.032918605146,  0.249351723343,  0.466911705055,  0.582030414044,  0.847252983783};

    const int s = 6;
    Vec w;
    Vec u(u0);
    double t;

#pragma unroll
    for (int i=0; i<s; i++)
    {
        t = t0 + dt*c[i];
        w = w*a[i] + Diff(u, t)*dt;
        u = u + w*b[i];
    }
    return u;
}

struct Vec4
{
    double y1, y2, y3, y4;

    Vec4(double y1=0, double y2=0, double y3=0, double y4=0) : y1(y1), y2(y2), y3(y3), y4(y4) {};

    Vec4 operator*(double v) const
    {
        return Vec4(y1*v, y2*v, y3*v, y4*v);
    }

    Vec4 operator+(const Vec4& v) const
    {
        return Vec4(y1+v.y1, y2+v.y2, y3+v.y3, y4+v.y4);
    }
};

struct CartPole
{
    const double mp = 0.1;
    const double mc = 1;
    const double l = 0.5;
    const double g = 9.81;
    int info, step;
    Vec4 u;
    double F, t;
		void reset(std::mt19937& gen)
		{
			#if SWINGUP
			    std::uniform_real_distribution<double> dist(-.1,.1);
			#else
			    std::uniform_real_distribution<double> dist(-0.05,0.05);
			#endif
			u = Vec4(dist(gen), dist(gen), dist(gen), dist(gen));
			F = t = step = 0;
			info = 1;
		}

		void getStateRew(vector<double>& state, double& rew)
		{
			state[0] = u.y1;
			state[1] = u.y2;
			state[2] = u.y4;
			state[3] = u.y3;
			state[4] = std::cos(u.y3);
			state[5] = std::sin(u.y3);
			#if SWINGUP
			double angle = std::fmod(u.y3, 2*M_PI);
			angle = angle<0 ? angle+2*M_PI : angle;
			rew = fabs(angle-M_PI)<M_PI/6 ? 1 : 0;
			#endif
		}

    Vec4 D(Vec4 u, double t)
    {
        Vec4 res;

        const double cosy = std::cos(u.y3);
        const double siny = std::sin(u.y3);
        const double w = u.y4;
#if SWINGUP
				const double fac1 = 1./(mc + mp * siny*siny);
				const double fac2 = fac1/l;
				res.y2 = fac1*(F + mp*siny*(l*w*w + g*cosy));
				res.y4 = fac2*(-F*cosy -mp*l*w*w*cosy*siny -(mc+mp)*g*siny);
#else
        const double fac1 = 1./(mp+mc);
        const double fac2 = l*(4./3. - fac1*(mp*cosy*cosy));
        const double F1 = F + mp * l * w * w * siny;
        res.y4 = (g*siny - fac1*F1*cosy)/fac2;
        res.y2 = fac1*(F1 - mp*l*res.y4*cosy);
#endif
        res.y1 = u.y2;
        res.y3 = u.y4;
        return res;
    }
};

Communicator * comm;
int main(int argc, const char * argv[])
{
  const int n = 1; //n agents
  //communication:
  const int sock = std::stoi(argv[1]);
  //time stepping
  const double dt = 4e-4;
	double rew = 0;
  std::mt19937 gen(sock);

  //communicator class, it needs a socket number sock, given by RL as first argument of execution
  Communicator comm(sock,6,1);
  //vector of state variables: in this case x, v, theta, ang_velocity
  vector<double> state(6);
  //vector of actions received by RL
  vector<double> actions(1);

  //random initial conditions:
  vector<CartPole> agents(n);
  for (auto& a : agents) a.reset(gen);

  while (true) {

    int k = 0; //agent ID, for now == 0
    for(auto& a : agents) {
			a.getStateRew(state, rew);

      comm.sendState(k, a.info, state, rew);
      comm.recvAction(actions);
      a.F = actions[0];
      a.info = 0; //at least one comm is done, so i set info to 0
			a.step++;
  		//advance the sim:
      for (int i=0; i<50; i++) {
        a.u = rk46_nl(a.t, dt, a.u, bind(&CartPole::D, &a, placeholders::_1, placeholders::_2));
        a.t += dt;

        //check if terminal state has been reached:
#if SWINGUP
        if(a.step>=500||(std::fabs(a.u.y1)>2.4)) {
#else
				if ((std::fabs(a.u.y3)>M_PI/15)||(std::fabs(a.u.y1)>2.4)) {
					rew = -1;
#endif
					a.getStateRew(state, rew);
          comm.sendState(k, 2, state, rew);
          a.reset(gen); //re-initialize (random initial conditions):
					rew = 0;
          break;
        }
      }
    }
  }
  return 0;
}
