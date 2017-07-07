//
//  main.cpp
//  acrobot
//
//  Created by Iveta Rott on January 7th, 2017 based on cart-pole.cpp by Dmitry Alexeev from 04/06/15.
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

struct Acrobot
{
  const double m1 = 1;    //mass arms
  const double m2 = 1;    //mass legs
  const double l1 = 1;    //length arms
  const double l2 = 1;    //length legs
  const double lc1 = 0.5; //length to center of mass of arms
  const double lc2 = 0.5; //length to center of mass of legs
  const double I1 = 1.;    //moment of inertia of arms // m1*l2/12
  const double I2 = 1.;    //moment of intertia of legs // m1*l1/12
  const double g = 9.81;  //gravity
  int info=1, step=0;
  double F=0, t=0;
  Vec4 u;

	void reset(std::mt19937& gen)
	{
		std::uniform_real_distribution<double> dist(-.1, .1);
		u = Vec4(dist(gen), dist(gen), dist(gen), dist(gen));
		//u.y1 = mapTheta1_to_2pi(u.y1);
		//u.y3 = mapTheta1_to_2pi(u.y3);
		F = t = step = 0;
		info = 1;
	}

	bool terminal() const
	{
		return cos(u.y1)+cos(u.y1+u.y3)<-1 && fabs(u.y2)<.5 && fabs(u.y4)<.5
	}

	void getState(vector<double>& state) const
	{
		state[0] = std::cos(u.y1);
		state[1] = std::sin(u.y1);
		state[2] = std::cos(u.y3);
		state[3] = std::sin(u.y3);
		state[4] = u.y2;
		state[5] = u.y4;
	}

  Vec4 D(Vec4 u, double t)
  {
    Vec4 res;

    const double cosy1 = cos(u.y1);
    const double siny1 = sin(u.y1);
    const double cosy2 = cos(u.y3);
    const double siny2 = sin(u.y3);

    const double d1 = m1*lc1*lc1 + m2*(l1*l1 + lc2*lc2 + 2.*l1*lc2*cosy2) + I1 + I2;
    const double d2 = m2*(lc2*lc2 + l1*lc2*cosy2) + I2;
    const double phi2 = m2*lc2*g*cos(u.y1 + u.y3 - M_PI/2.);
    const double phi1 = -m2*l1*lc2*u.y4*u.y4*siny2 - 2.*m2*l1*lc2*u.y4*u.y2*siny2 + (m1*lc1+m2*l1)*g*cos(u.y1-M_PI/2.) + phi2;

    res.y4 = 1./(m2*lc2*lc2+I2-d2*d2/d1)*(F + d2/d1*phi1 - m2*l1*lc2*u.y2*u.y2*siny2-phi2);
    res.y2 = -1./d1 * (d2*res.y4 + phi1);
    res.y1 = u.y2;
    res.y3 = u.y4;
    return res;
  }
};

inline double mapTheta1_to_2pi(double theta)
{
 theta = fmod(theta, 2*M_PI); //theta between -2pi and 2pi
 if(theta<0) theta += 2*M_PI; //theta between 0 and 2*pi
 return theta;
}

Communicator * comm;
int main(int argc, const char * argv[])
{
  const int n = 1, sock = std::stoi(argv[1]);
  //time stepping
  const double dt = 1e-3;
  std::mt19937 gen(sock);
  Communicator comm(sock, 6, 1);
  vector<double> state(6), actions(1);

  //random initial conditions:
  vector<Acrobot> agents(n);
  for (auto& a : agents) a.reset(gen);

  while (true) {
    int k = 0; //agent ID, for now == 0
    for (auto& a : agents) {
	    a.getState(state);
			comm.sendState(k, a.info, state, 0);
	    comm.recvAction(actions);
	    a.F = actions[0];
	    a.info = 0; //at least one comm is done, so i set info to 0
			a.step++;

	    for (int i=0; i<200; i++) {
	      a.u = rk46_nl(a.t, dt, a.u, bind(&Acrobot::D, &a, placeholders::_1, placeholders::_2));
	      a.t += dt;

	      if (a.terminal() || a.step >= 500) {
	        a.getState(state);
	        comm.sendState(k, 2, state, a.terminal() ? 1 : 0);
	        //re-initialize the simulations (random initial conditions):
		    	a.reset(gen);
	        break;
	      }
		  }
		}
	}
  return 0;
}
