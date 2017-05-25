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
//#define __SMART_
//#define __PRINT_
#define TERM_REW_FAC 50
#define HEIGHT_PENAL 10
using namespace std;

struct Vec7
{
    double u,v,w,x,y,a,T,E; //x vel, y vel, ang vel, x pos, y pos, angle (or their time derivatives)

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

    Vec7(const double _u=0, const double _v=0, const double _w=0, const double _x=0,
      const double _y=0, const double _a=0, const double _T=0, const double _E=0)
    : u(_u), v(_v), w(_w), x(_x), y(_y), a(_a), T(_T), E(_E) {};

    Vec7(const Vec7& c) :
      u(c.u), v(c.v), w(c.w), x(c.x), y(c.y), a(c.a), T(c.T), E(c.E) { }

    Vec7 operator*(double s) const
    {
        return Vec7(u*s, v*s, w*s, x*s, y*s, a*s, T*s, E*s);
    }

    Vec7 operator+(const Vec7& s) const
    {
        return Vec7(u+s.u, v+s.v, w+s.w, x+s.x, y+s.y, a+s.a, T+s.T, E+s.E);
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

    #pragma unroll
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
    const double pi  = 3.14159265359;
    const double CT  = 1.2;
    const double Aa  = 1.4;
    const double Bb  = 1.0;
    const double mut = 0.20;
    const double nut = 0.20;
    const double piinv = 1/3.14159265359;
    const double CR = 3.14159265359;

    const double II  = 20.0;
    const double beta= 0.1;
    //const double II  = 3.; //we have to multiply *2 all moments by Anderson
    //const double beta= 0;

    double Jerk, oldDistance, oldTorque, oldAngle, oldEnergySpent, time; //angular jerk
    int info;
    Vec7 _s;
    Glider(): Jerk(0), time(0), info(1), oldAngle(0), oldTorque(0), oldEnergySpent(0), oldDistance(0) {}

    void reset()
    {
      info=1; oldDistance=0; oldEnergySpent=0; Jerk=0; time=0; oldTorque=0;
    }

    Vec7 D(const Vec7& s, const double t)
    {
        Vec7 res;
        const double eps = 2.2e-16;
        const double uv2p = s.u*s.u + s.v*s.v;
        const double suv2 = sqrt(uv2p);
        const double uv2n = s.u*s.u - s.v*s.v;
        const double _f1  = s.u*s.v/(suv2+eps);
        const double _f2  = uv2n/(uv2p+eps);
        const double Gamma = (2*piinv)*(-CT*_f1 + CR*s.w);
        const double F = piinv*(Aa - Bb*_f2)*suv2*s.u;
        const double G = piinv*(Aa - Bb*_f2)*suv2*s.v;
        const double M = (mut + nut*std::fabs(s.w))*s.w;

        const double sinth = std::sin(s.a);
        const double costh = std::cos(s.a);
        const double betasq= beta*beta;
        const double fact1 = II + betasq;
        const double fact2 = II + 1.;
        const double fact3 = .25*(II*(1+betasq)+.5*std::pow(1-betasq,2));

        res.u = ( fact2*s.v*s.w - Gamma*s.v - sinth - F)/fact1;
        res.v = (-fact1*s.u*s.w + Gamma*s.u - costh - G)/fact2;
        res.w = ((betasq-1.0)*s.u*s.v + s.T - M)/fact3;
        res.x = s.u*costh - s.v*sinth;
        res.y = s.u*sinth + s.v*costh;
        res.a = s.w;
        res.T = Jerk;
        res.E = s.T*s.T; //using performance metric eq 4.9 instead of energy

        return res;
    }

    void print(const int ID) const
    {
        string suff = "trajectory_" + to_string(ID) + ".dat";
        FILE * f = fopen(suff.c_str(), time==0 ? "w" : "a");
        fprintf(f,"%e %e %e %e %e %e %e %e %e\n",
                time,_s.u,_s.v,_s.w,_s.x,_s.y,_s.a,_s.T,_s.E);
        fclose(f);
    }

    double getDistance() const
    {
      //goal is {100,-50}
      const double xx = std::pow(_s.x-100.,2);
      const double yy = 0;//std::pow(_s.y+ 50.,2);
      return std::sqrt(xx+yy);
    }

    void prepareState(vector<double>& state) const
    {
      assert(state.size() == 9);
      state[0] = _s.u;
      state[1] = _s.v;
      state[2] = _s.w;
      state[3] = _s.x;
      state[4] = _s.y;
      state[5] = _s.a;
      state[6] = _s.T;
      state[7] = _s.vx();
      state[8] = _s.vy();
    }

    void updateOldDistanceAndEnergy()
    {
      oldDistance = getDistance();
      oldEnergySpent = _s.E;
      //same time, put angle back in 0, 2*pi
      _s.a = std::fmod( _s.a, 2*M_PI);
      _s.a = _s.a < 0 ? _s.a +2*M_PI : _s.a;
      oldAngle = _s.a;
      oldTorque = _s.T;
    }
};
#ifdef __SMARTIES_
Communicator * comm;
#endif
int main(int argc, const char * argv[])
{
    const int n = 1; //n agents
    int k = 0;
    const double eps  = 2.2e-16;
    //time stepping
    const double dt = 1e-3;
    const double tol_pos = 1.0;
    const double tol_ang = 0.1;
    const int nstep = 500;
    #ifdef __SMARTIES_
    //communication:
    const int sock = std::stoi(argv[1]);
    std::mt19937 gen(sock);
    std::uniform_real_distribution<double> d1(-.1,.1); //to be used for vels, angle
    std::uniform_real_distribution<double> d2(-1.,1.); //to be used for position
    Communicator comm(sock,9,1);
    #endif
    vector<double> state(9);
    vector<double> actions(1);

    //random initial conditions:
    vector<Glider> agents(n);
    for (auto& a : agents) {
      #ifdef __SMARTIES_ //u,v,w,x,y,a,T
        a._s = Vec7(d1(gen), d1(gen), d1(gen), d2(gen), d2(gen), d1(gen), 0, 0);
      #else
        //a._s = Vec7(0, 0, 0, 0, 0, 0, 1.,    0);
        //a._s = Vec7(0, 0, 0, 0, 0, 0, 0.105, 0);
        a._s = Vec7(0.0001, 0.0001, 0.0001, 0, 0, 0, 0.0001,     0);
      #endif
      #ifdef __PRINT_
                  a.print(k);
      #endif
        a.updateOldDistanceAndEnergy();
    }

    while (true) {

        //const int k = 0; //agent ID, for now == 0
        for (auto& a : agents) { //assume we have only one agent per application for now...
            const double dist_gain = a.oldDistance - a.getDistance();
            const double rotation = std::fabs(a.oldAngle-a._s.a);
            const double jerk = std::fabs(a.oldTorque - a._s.T);
            const double performamce = a._s.E - a.oldEnergySpent + eps;
            //reward clipping: what are the proper scaled? TODO
            //const double reward=std::min(std::max(-1.,dist_gain/performamce),1.);
#if 1
            double reward = dist_gain;
#else
            double reward = dist_gain/performamce; //JUST NOPE
#endif
#if 1
            reward -= rotation;
#endif
#if 1
            reward -= performamce;
#endif
#if 1
            reward -= jerk;       
#endif
            a.updateOldDistanceAndEnergy();
            //load state:
            a.prepareState(state);

#ifdef __SMARTIES_
            comm.sendState(0, a.info, state, reward);
            comm.recvAction(actions);
            //a.Jerk = actions[0];
            a._s.T = actions[0];
            a.info = 0; //at least one comm is done, so i set info to 0
#endif

        	//advance the sim:
            for (int i=0; i<nstep; i++) {
                a._s = rk46_nl(a.time, dt, a._s, bind(&Glider::D, &a, placeholders::_1, placeholders::_2));
                a.time += dt;

                //check if terminal state has been reached:
                const bool max_torque = std::fabs(a._s.T)>5;
                const bool way_too_far = a._s.x > 125;
                const bool hit_bottom = a._s.y <= -50;
                const bool wrong_xdir = a._s.x < -10;
                const bool timeover = a.time > 2000;
#ifdef __SMARTIES_
                if (max_torque || hit_bottom || wrong_xdir || way_too_far )
                {
                    a.info = 2; //tell RL we are in terminal state
                    const bool got_there = a.getDistance() < tol_pos;
                    const bool landing = std::fabs(a._s.a - .25*M_PI) < tol_ang;
                    double final_reward = 0;
                    //these rewards will then be multiplied by 1/(1-gamma)
                    //in RL algorithm, so that internal RL scales make sense
                    final_reward  = got_there ? TERM_REW_FAC : -a.getDistance();
                    final_reward += (landing && got_there) ? TERM_REW_FAC : 0;

		               if(wrong_xdir || max_torque || way_too_far) 
			               final_reward = -100 -HEIGHT_PENAL*pow(50+a._s.y,2); 

                    a.prepareState(state);
                    //printf("Sending term state %f %f %f %f\n",
                    //state[0],state[1],state[2],state[3]); 
                    //fflush(0);
                    comm.sendState(0, a.info, state, final_reward);

                    a.reset(); //set info back to 0
                    a._s = Vec7(d1(gen), d1(gen), d1(gen), d2(gen), d2(gen), d1(gen), 0, 0);
                    a.updateOldDistanceAndEnergy();
                    #ifdef __PRINT_
                      a.print(k);
                    #endif
                    k++;
                    break;
                }
#else
                if (a._s.y <= -500 || timeover)
                {
                    a.print(k);
                    return 0;
                }
#endif
            }
#ifdef __PRINT_
            a.print(k);
#endif
        }
    }

    return 0;
}
