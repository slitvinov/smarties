//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <functional>
#include "communicator.h"

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
    int info;
    Vec4 u;
    double F;
    
    Vec4 D(Vec4 u, double t)
    {
        Vec4 res;
        
        const double cosy = cos(u.y3);
        const double siny = sin(u.y3);
        
        const double fac1 = 1./(mp+mc);
        const double fac2 = l*(4./3. - fac1*(mp*cosy*cosy));
        
        const double F1 = F + mp * l * u.y4 * u.y4 * siny;
        
        res.y4 = (g*siny - fac1*F1*cosy)/fac2;
        res.y2 = fac1*(F1 - mp*l*res.y4*cosy);
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
    const int nssteps = 1;
    const double dt = 2e-2;
    double t = 0;
    //trash:
    long long int nfallen(0), sincelast(0), duringlast(0), ntot(0);
    double percfallen = 0.0;
    
    comm = new Communicator(sock,4,1);
    vector<double> state(4), actions(1);
    
    vector<CartPole> agents(n);
    for (auto& a : agents) {
        a.u = Vec4( .01*(drand48()-.5), .01*(drand48()-.5), .01*(drand48()-.5), .01*(drand48() -.5));
        a.F    = 0;
        a.info = 1;
    }
    while (true) {
        
        int k(0); //agent ID, for now == 0
        for (auto& a : agents) {
            double r = 0.;
            //ntot += 1; sincelast += 1;
            state[0] = a.u.y1;
            state[1] = a.u.y2;
            state[2] = a.u.y4;
            state[3] = a.u.y3;
            
            //printf("Sending state %f %f %f %f\n",state[0],state[1],state[2],state[3]); fflush(0);
            comm->sendState(k, a.info, state, r);
            comm->recvAction(actions);
            a.F = actions[0];
            a.info = 0;
            //printf("Received action %f\n", a.F); fflush(0);
        }
        
        
        for (int i=0; i<nssteps; i++) {
            for (auto& a : agents) {
                a.u = rk46_nl(t, dt, a.u, bind(&CartPole::D, &a, placeholders::_1, placeholders::_2));
                
                if ((fabs(a.u.y3)>.2)||(fabs(a.u.y1)>2.4)) {
                    //nfallen += 1; sincelast = 0; percfallen = nfallen/ntot;
                    
                    a.info = 2;
                    double r = -1.;
                    state[0] = a.u.y1;
                    state[1] = a.u.y2;
                    state[2] = a.u.y4;
                    state[3] = a.u.y3;
                    //printf("Sending term state %f %f %f %f\n",state[0],state[1],state[2],state[3]); fflush(0);
                    comm->sendState(k, a.info, state, r);
                    
                    a.u = Vec4( .01*(drand48()-.5),
                                .01*(drand48()-.5),
                                .01*(drand48()-.5),
                                .01*(drand48()-.5));
                    t = 0;
                    a.F = 0;
                    a.info = 1;
                }
            }
            
            t += dt;
        }
        /*
        if (ntot % 10000 == 0) {
            cout << nfallen - duringlast << endl;
            duringlast =+ nfallen;
        }
         */
    }
    
    return 0;
}
