#include "smarties.h"
#include <iostream>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>
#include <functional>
#define SWINGUP 0

template <typename Func, typename Vec>
Vec rk46_nl(double t0, double dt, Vec u0, Func&& Diff)
{
  static constexpr double a[] = {0.000000000000, -0.737101392796,
            -1.634740794341, -0.744739003780, -1.469897351522, -2.813971388035};
  static constexpr double b[] = {0.032918605146,  0.823256998200,
             0.381530948900,  0.200092213184,  1.718581042715,  0.270000000000};
  static constexpr double c[] = {0.000000000000,  0.032918605146,
             0.249351723343,  0.466911705055,  0.582030414044,  0.847252983783};
  static constexpr int s = 6;
  Vec w;
  Vec u(u0);
  double t;

  for (int i=0; i<s; ++i)
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

  Vec4(double _y1=0, double _y2=0, double _y3=0, double _y4=0) :
    y1(_y1), y2(_y2), y3(_y3), y4(_y4) {};

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
  const double dt = 4e-4;
  const int nsteps = 50;
  // const double dt = 1e-6;   // emulate expensive application
  // const int nsteps = 20000; //
  int step=0;
  Vec4 u;
  double F=0, t=0;

	void reset(std::mt19937& gen)
	{
		#if SWINGUP
	    std::uniform_real_distribution<double> dist(-.1,.1);
		#else
	    std::uniform_real_distribution<double> dist(-0.05,0.05);
		#endif
		u = Vec4(dist(gen), dist(gen), dist(gen), dist(gen));
    step = 0;
		F = 0;
    t = 0;
	}

  bool is_failed()
  {
    #if SWINGUP
      return std::fabs(u.y1)>2.4;
    #else
      return std::fabs(u.y1)>2.4 || std::fabs(u.y3)>M_PI/15;
    #endif
  }
  bool is_over()
  {
    #if SWINGUP
      return step>=500 || std::fabs(u.y1)>2.4;
    #else
      return step>=500 || std::fabs(u.y1)>2.4 || std::fabs(u.y3)>M_PI/15;
    #endif
  }

  int advance(std::vector<double> action)
  {
    F = action[0];
    step++;
    for (int i=0; i<nsteps; i++) {
      u = rk46_nl(t, dt, u, std::bind(&CartPole::Diff,
                                      this,
                                      std::placeholders::_1,
                                      std::placeholders::_2) );
      t += dt;
      if( is_over() ) return 1;
    }
    return 0;
  }

	std::vector<double> getState(const int size = 6)
	{
    assert(size == 4 || size == 6);
    std::vector<double> state(size);
		state[0] = u.y1;
		state[1] = u.y2;
		state[2] = u.y4;
		state[3] = u.y3;
    if(size == 6) {
      state[4] = std::cos(u.y3);
      state[5] = std::sin(u.y3);
    }
		return state;
	}

  double getReward()
  {
    #if SWINGUP
      double angle = std::fmod(u.y3, 2*M_PI);
      angle = angle<0 ? angle+2*M_PI : angle;
      return std::fabs(angle-M_PI)<M_PI/6 ? 1 : 0;
    #else
      //return -1*( fabs(u.y3)>M_PI/15 || fabs(u.y1)>2.4 );
      return 1 - ( std::fabs(u.y3)>M_PI/15 || std::fabs(u.y1)>2.4 );
    #endif
  }

  Vec4 Diff(Vec4 _u, double _t)
  {
    Vec4 res;

    const double cosy = std::cos(_u.y3), siny = std::sin(_u.y3);
    const double w = _u.y4;
    #if SWINGUP
      const double fac1 = 1/(mc + mp * siny*siny);
      const double fac2 = fac1/l;
      res.y2 = fac1*(F + mp*siny*(l*w*w + g*cosy));
      res.y4 = fac2*(-F*cosy -mp*l*w*w*cosy*siny -(mc+mp)*g*siny);
    #else
      const double totMass = mp+mc;
      const double fac2 = l*(4.0/3 - (mp*cosy*cosy)/totMass);
      const double F1 = F + mp * l * w * w * siny;
      res.y4 = (g*siny - F1*cosy/totMass)/fac2;
      res.y2 = (F1 - mp*l*res.y4*cosy)/totMass;
    #endif
    res.y1 = _u.y2;
    res.y3 = _u.y4;
    return res;
  }
};

inline void app_main(smarties::Communicator*const comm, int argc, char**argv)
{
  const int control_vars = 1; // force along x
  const int state_vars = 6; // x, vel, angvel, angle, cosine, sine
  comm->setStateActionDims(state_vars, control_vars);

  //OPTIONAL: action bounds
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm->setActionScales(upper_action_bound, lower_action_bound, bounded);

  /*
    // ALTERNATIVE for discrete actions:
    vector<int> n_options = vector<int>{2};
    comm->set_action_options(n_options);
    // will receive either 0 or 1, app chooses resulting outcome
  */

  //OPTIONAL: hide state variables. e.g. show cosine/sine but not angle
  std::vector<bool> b_observable = {true, true, true, false, true, true};
  //std::vector<bool> b_observable = {true, false, false, false, true, true};
  comm->setStateObservable(b_observable);
  //comm->setIsPartiallyObservable();

  CartPole env;

  while(true) //train loop
  {
    env.reset(comm->getPRNG()); // prng with different seed on each process
    comm->sendInitState(env.getState()); //send initial state

    while (true) //simulation loop
    {
      std::vector<double> action = comm->recvAction();
      if(comm->terminateTraining()) return; // exit program

      bool poleFallen = env.advance(action); //advance the simulation:

      std::vector<double> state = env.getState();
      double reward = env.getReward();

      if(poleFallen) { //tell smarties that this is a terminal state
        comm->sendTermState(state, reward);
        break;
      } else comm->sendState(state, reward);
    }
  }
}

int main(int argc, char**argv)
{
  smarties::Engine e(argc, argv);
  if( e.parse() ) return 1;
  e.run( app_main );
  return 0;
}
