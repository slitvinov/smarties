#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <cassert>
#include <random>
#include <functional>
#ifdef __SMARTIES_
#include "Communicator.h"
#endif

//#define __PRINT_
//#define SPEED_PENAL
#define RANDOM_START 1
//#define NOISY 0

#ifndef RHORATIO
#define RHORATIO 200
#endif
#ifndef ASPECTRATIO
#define ASPECTRATIO 0.1
#endif
#ifndef INSTREW
#define INSTREW 1 /* 0 is default, 1 is time optimal, 2 is energy optimal */
#endif

#if   INSTREW==1
#define TERM_REW_FAC 200
#elif INSTREW==0
#define TERM_REW_FAC 100
#else
#define TERM_REW_FAC 50
#endif
using namespace std;

struct Vec7
{
  double u,v,w,x,y,a;//,T,E; //x vel, y vel, ang vel, x pos, y pos, angle (or their time derivatives)

  double vx() const
  {
      const double sina = std::sin(a);
      const double cosa = std::cos(a);
      return u*cosa + v*sina;
  }

  double vy() const
  {
      const double sina = std::sin(a);
      const double cosa = std::cos(a);
      return v*cosa - u*sina;
  }

  Vec7(const double _u=0,const double _v=0,const double _w=0,const double _x=0,
    const double _y=0,const double _a=0,const double _T=0,const double _E=0)
    : u(_u), v(_v), w(_w), x(_x), y(_y), a(_a) {}

  Vec7(const Vec7& c) : u(c.u), v(c.v), w(c.w), x(c.x), y(c.y), a(c.a) {}

  Vec7 operator*(double s) const
  {
      return Vec7(u*s, v*s, w*s, x*s, y*s, a*s);//, T*s, E*s);
  }

  Vec7 operator+(const Vec7& s) const
  {
      return Vec7(u+s.u, v+s.v, w+s.w, x+s.x, y+s.y, a+s.a);//, T+s.T, E+s.E);
  }
};

/*
 Julien Berland, Christophe Bogey, Christophe Bailly,
 Low-dissipation and low-dispersion fourth-order Runge-Kutta algorithm,
 Computers & Fluids, Volume 35, Issue 10, December 2006, Pages 1459-1463, ISSN 0045-7930,
 http://dx.doi.org/10.1016/j.compfluid.2005.04.003
 */

template <typename Func, typename Vec>
Vec rk46_nl(double t0, double dt, const Vec u0, Func&& Diff)
{
  const double a[] = {0.000000000000, -0.737101392796, -1.634740794341, -0.744739003780, -1.469897351522, -2.813971388035};
  const double b[] = {0.032918605146,  0.823256998200,  0.381530948900,  0.200092213184,  1.718581042715,  0.270000000000};
  const double c[] = {0.000000000000,  0.032918605146,  0.249351723343,  0.466911705055,  0.582030414044,  0.847252983783};

  const int s = 6;
  Vec w; // w(t0) = 0
  Vec u(u0);
  double t;

  for (int i=0; i<s; i++) {
      t = t0 + dt*c[i];
      w = w*a[i] + Diff(u, t)*dt;
      u = u + w*b[i];
  }
  return u;
}

/*
 P. Paoletti, and L. Mahadevan,
 Planar controlled gliding, tumbling and descent,
 Journal of Fluid Mechanics 689 (2011): 489-516.
 http://dx.doi.org/10.1017/jfm.2011.426
 also see
 A. Andersen, U. Pesavento, and Z. Wang.
 Analysis of transitions between fluttering, tumbling and steady descent of falling cards,
 Journal of Fluid Mechanics 541 (2005): 91-104.
 http://dx.doi.org/10.1017/S0022112005005847
 */

struct Glider
{
  const double CT  = 1.2, Aa  = 1.4, Bb  = 1, mut = .2, nut = .2, CR = M_PI;
  const double II  = RHORATIO * ASPECTRATIO;
  const double beta = ASPECTRATIO;

  //time stepping
  const double dt = 5e-4;
  const int nstep = 500;
  const double DT = dt*nstep;

  double Jerk=0, Torque=0, oldDistance=0, oldTorque=0;
  double oldAngle=0, oldEnergySpent=0, time=0;
  double startx = -20; // used only for area bombing

  int info=1, step=0;
  Vec7 _s;

  void reset(std::mt19937& gen)
  {
    info=1;
    std::uniform_real_distribution<double> init(-.1,.1); //for vels, angle
    std::uniform_real_distribution<double> initx(-10,10); //for position
    Jerk=Torque=oldDistance=oldTorque=oldAngle=oldEnergySpent=time=step=0;

    #ifdef __SMARTIES_ //u,v,w,x,y,a,T
      #if RANDOM_START == 1
        _s = Vec7(init(gen), init(gen), 0, initx(gen), 0, init(gen), 0, 0);
      #elif RANDOM_START == 2
        _s = Vec7(0, 0, 0, startx, 0, 0, 0, 0);
      #else
        _s = Vec7(0, 0, 0, 0, 0, 0, 0, 0);
      #endif
    #else
      _s = Vec7(0.0001, 0.0001, 0.0001, 0, 0, 0, 0.0001,     0);
    #endif

    #ifdef __PRINT_
      print(0);
    #endif
    updateOldDistanceAndEnergy();
  }

  bool is_over() const
  {
    //const bool max_torque = std::fabs(a.Torque)>5;
    const bool way_too_far = _s.x > 200;
    const double slack = 0.4*std::max(0., std::min(_s.x-50, 100-_s.x));
    const bool hit_bottom =  _s.y <= -50 -slack;
    const bool wrong_xdir = _s.x < -50;
    //const bool timeover = time > 1000;

    return ( hit_bottom || wrong_xdir || way_too_far );
  }

  int advance(vector<double> action)
  {
    updateOldDistanceAndEnergy();
    Torque = action[0];
    info = 0; // i received an action!
    step++;
    for (int i=0; i<nstep; i++) {
      _s = rk46_nl(time, dt, _s, bind(&Glider::Diff, this, placeholders::_1, placeholders::_2) );
      time += dt;
      if( is_over() ) {
        info = 2;
        return 1;
      }
    }
    return 0;
  }

  vector<double> getState(std::mt19937& gen)
  {
    vector<double> state(10);
    state[0] = _s.u;
    state[1] = _s.v;
    state[2] = _s.w;
    state[3] = _s.x;
    state[4] = _s.y;
    state[5] = std::cos(_s.a);
    state[6] = std::sin(_s.a);
    state[7] = Torque;
    state[8] = _s.vx();
    state[9] = _s.vy();
    #ifdef NOISY
      std::normal_distribution<double> noise(0,.1);
      state[0] += noise(gen);
      state[1] += noise(gen);
      state[2] += noise(gen);
    #endif
    return state;
  }

  double getReward()
  {
    const double dist_gain = oldDistance - getDistance();
    const double rotation = std::fabs(oldAngle - _s.a);
    const double jerk = std::fabs(oldTorque - Torque)/DT;
    //const double performamce =  std::fabs(a.Torque);//a._s.E - a.oldEnergySpent + eps;
    const double performamce = std::pow(Torque, 2);
    //reward clipping: what are the proper scaled? TODO
    //const double reward=std::min(std::max(-1.,dist_gain/performamce),1.);

    #if INSTREW==0
      const double reward = dist_gain -performamce -jerk;
    #elif INSTREW==1
      const double reward = dist_gain -DT;
    #else
      const double reward = dist_gain -performamce;
    #endif

    #ifdef __PRINT_
      a.print(k);
    #endif
    return reward;
  }

  double getTerminalReward()
  {
    info = 2; //tell RL we are in terminal state
    _s.a = std::fmod(_s.a, 2*M_PI);
    _s.a = _s.a<0 ? _s.a+2*M_PI : _s.a;
    //these rewards will then be multiplied by 1/(1-gamma)
    //in RL algorithm, so that internal RL scales make sense
    const double dist = getDistance(), rela = std::fabs(_s.a -.25*M_PI);
    const double xrew = dist>5 ? 0 : std::exp(-dist*dist);
    const double arew = (rela>M_PI/4||dist>5) ? 0 : std::exp(-10*rela*rela);
    double final_reward  = TERM_REW_FAC*(xrew + arew);

    #ifdef SPEED_PENAL
    {
      if (std::fabs(_s.u) > 0.5)
        final_reward *= std::exp(-10*std::pow(std::fabs(_s.u)-.5,2));
      if (std::fabs(_s.v) > 0.5)
        final_reward *= std::exp(-10*std::pow(std::fabs(_s.v)-.5,2));
      if (std::fabs(_s.w) > 0.5)
        final_reward *= std::exp(-10*std::pow(std::fabs(_s.w)-.5,2));
    }
    #endif

    return final_reward - dist;
  }

  Vec7 Diff(const Vec7& s, const double t)
  {
    Vec7 res;
    const double eps = 2.2e-16;
    const double uv2p = s.u*s.u + s.v*s.v;
    const double uv2n = s.u*s.u - s.v*s.v;
    const double _f1  = s.u*s.v/(std::sqrt(uv2p)+eps);
    const double _f2  = uv2n/(uv2p+eps);
    const double G = (2/M_PI)*(CR*s.w -CT*_f1);
    const double F = (1/M_PI)*(Aa -Bb*_f2)*std::sqrt(uv2p);
    const double M = (mut + nut*std::fabs(s.w))*s.w;

    const double sinth = std::sin(s.a), costh = std::cos(s.a);
    const double betasq= beta*beta;
    const double fact1 = II + betasq;
    const double fact2 = II + 1.;
    const double fact3 = .25*(II*(1+betasq)+.5*std::pow(1-betasq,2));

    res.u = ( fact2*s.v*s.w - G*s.v - sinth - F*s.u)/fact1;
    res.v = (-fact1*s.u*s.w + G*s.u - costh - F*s.v)/fact2;
    res.w = ((betasq-1.0)*s.u*s.v + Torque - M)/fact3;
    res.x = s.u*costh - s.v*sinth;
    res.y = s.u*sinth + s.v*costh;
    res.a = s.w;
    //res.T = Jerk;
    //res.E = .5*s.T*s.T; //using performance metric eq 4.9 instead of energy

    return res;
  }

  void print(const int ID) const
  {
      string suff = "trajectory_" + to_string(ID) + ".dat";
      FILE * f = fopen(suff.c_str(), time==0 ? "w" : "a");
      fprintf(f,"%e %e %e %e %e %e %e\n",
              time,_s.u,_s.v,_s.w,_s.x,_s.y,_s.a);//,_s.T,_s.E);
      fclose(f);
  }

  double getDistance() const
  {
    //goal is {100,-50}
    const double xx = std::pow(_s.x-100,2);
    const double yy = 0;//std::pow(_s.y+ 50.,2);
    return std::sqrt(xx+yy);
  }

  void updateOldDistanceAndEnergy()
  {
    oldDistance = getDistance();
    oldEnergySpent += DT*Torque*Torque;//_s.E;
    //same time, put angle back in 0, 2*pi
    _s.a = std::fmod( _s.a, 2*M_PI);
    _s.a = _s.a < 0 ? _s.a +2*M_PI : _s.a;
    oldAngle = _s.a;
    oldTorque = Torque;
  }
};

int main(int argc, const char * argv[])
{
  std::cout << "Glider with density ratio " << RHORATIO <<
   " and aspect ratio = " << ASPECTRATIO << ". Instantaneous reward is " <<
   (INSTREW == 0 ? "mixed" : (INSTREW == 1 ? "time" : "energy"));
  #ifdef SPEED_PENAL
  std::cout << " with penalization on terminal velocity";
  #endif
  std::cout << "." << std::endl;
  #ifdef __SMARTIES_
    //communication:
    const int sock = std::stoi(argv[1]);
    std::mt19937 gen(sock);
    Communicator comm(sock,10,1);

    bool bounded = true;
    vector<double> upper_action_bound{1}, lower_action_bound{-1};
    comm.set_action_scales(upper_action_bound, lower_action_bound, bounded);
    vector<bool> b_observable = {1, 1, 1, 1, 1, 1, 1, 0, 0, 0};
    comm.set_state_observable(b_observable);
  #endif

  //random initial conditions:
  Glider env;

  while (true) //train loop
  {
    //reset environment:
    env.reset(gen); //comm contains rng with different seed on each rank
    #ifdef __SMARTIES_
      //send initial state:
      comm.sendInitState(env.getState(comm.gen));
    #endif

    while (true) //simulation loop
    {
      #ifdef __SMARTIES_
        vector<double> action = comm.recvAction();
      #else
        vector<double> action = {0};
      #endif

      //advance the simulation:
      bool terminated = env.advance(action);

      vector<double> state = env.getState(gen);
      double reward = env.getReward();

      if(terminated)  //tell smarties that this is a terminal state
      {
        #ifdef __SMARTIES_
          comm.sendTermState(state, env.getTerminalReward());
        #endif

        break;
      }
      #ifdef __SMARTIES_
      else comm.sendState(state, reward);
      #endif
    }
  }
  return 0;
}
