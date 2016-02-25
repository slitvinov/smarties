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
#include "Util/communicator.h"
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

struct state
{
    vector<vector<int>> s;
    state()
    {
        vector<int> t(1);
        t[0]=75;
        s.push_back(tmp);
        t[0]=95; t.push_back(64);
        s.push_back(tmp);
        t[0]=17; t[0]=47; t.push_back(82);
        s.push_back(tmp);
        t[0]=18; t[0]=35; t[0]=87; t.push_back(10);
        s.push_back(tmp);
        t[0]=20; t[0]=04; t[0]=82; t[0]=47; t.push_back(65);
        s.push_back(tmp);
        t[0]=19; t[0]=01; t[0]=23; t[0]=75; t[0]=03; t.push_back(34);
        s.push_back(tmp);
        t[0]=88; t[0]=02; t[0]=77; t[0]=73; t[0]=07; t[0]=63; t.push_back(67);
        s.push_back(tmp);
        t[0]=99; t[0]=65; t[0]=04; t[0]=28; t[0]=06; t[0]=16; t[0]=70; t.push_back(92);
        s.push_back(tmp);
        t[0]=41; t[0]=41; t[0]=26; t[0]=56; t[0]=83; t[0]=40; t[0]=80; t[0]=70; t.push_back(33);
        s.push_back(tmp);
        t[0]=41; t[0]=48; t[0]=72; t[0]=33; t[0]=47; t[0]=32; t[0]=37; t[0]=16; t[0]=94; t.push_back(29);
        s.push_back(tmp);
        t[0]=53; t[0]=71; t[0]=44; t[0]=65; t[0]=25; t[0]=43; t[0]=91; t[0]=52; t[0]=97; t[0]=51; t.push_back(14);
        s.push_back(tmp);
        t[0]=70; t[0]=11; t[0]=33; t[0]=28; t[0]=77; t[0]=73; t[0]=17; t[0]=78; t[0]=39; t[0]=68; t[0]=17; t.push_back(57);
        s.push_back(tmp);
        t[0]=91; t[0]=71; t[0]=52; t[0]=38; t[0]=17; t[0]=14; t[0]=91; t[0]=43; t[0]=58; t[0]=50; t[0]=27; t[0]=29; t.push_back(48);
        s.push_back(tmp);
        t[0]=63; t[0]=66; t[0]=04; t[0]=68; t[0]=89; t[0]=53; t[0]=67; t[0]=30; t[0]=73; t[0]=16; t[0]=69; t[0]=87; t[0]=40; t.push_back(31);
        s.push_back(tmp);
        t[0]=04; t[0]=62; t[0]=98; t[0]=27; t[0]=23; t[0]=09; t[0]=70; t[0]=98; t[0]=73; t[0]=93; t[0]=38; t[0]=53; t[0]=60; t[0]=04; t.push_back(23);
        s.push_back(tmp);
    }
};
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
    const double m = 1;
    const double M = 2;
    const double l = 0.5;
    const double g = 9.81;
    
    Vec4 u;
    double F;
    
    Vec4 D(Vec4 u, double t)
    {
        Vec4 res;
        
        const double cosy = cos(u.y3);
        const double siny = sin(u.y3);
        const double tmp1 = 1.0 / (M+m*(1-cosy));
        const double tmp2 = u.y4*u.y4*l;
        
        res.y1 = u.y2;
        res.y2 = m * tmp1 * ((tmp2 - cosy*g)*siny + F/m);
        res.y3 = u.y4;
        res.y4 = tmp1/l * (siny * (g*(m+M) - m*tmp2) - F);
        return res;
    }
};

Communicator * comm;
int main(int argc, const char * argv[])
{
    const int n = 1;
    long long int nfallen = 0;
    long long int sincelast = 0;
    long long int duringlast = 0;
    long long int ntot = 0;
    double percfallen = 0.0;
    const int nssteps = 20;
    comm = new Communicator(4,n);
    vector<double> state(4);
    const double dt = 1e-4;
    double t = 0;
    bool first = true;
    vector<CartPole> agents(n);
    for (auto& a : agents)
    {
        a.u = Vec4( .02*(drand48()-.5), .02*(drand48()-.5), .02*(drand48()-.5), .02*(drand48() -.5));
        a.F    = 0;
    }
    
    while (ntot<=1250000)
    {
        if (first)
        {
            printf("Initial state! \n");
            for (auto& a : agents)
            {
                state[0] = a.u.y1;
                state[1] = a.u.y2;
                state[2] = a.u.y4;
                state[3] = a.u.y3;
                comm->sendState(state, 0.0);
            }
            first = false;
        }
        
        int act;
        printf("Receiving action! \n");
        for (auto& a : agents)
        {
            act = comm->recvAction();
            switch (act)
            {
                case 0:
                    a.F = -50.;
                    break;
                case 1:
                    a.F = -5.;
                    break;
                case 2:
                    a.F =  0.;
                    break;
                case 3:
                    a.F =  5.;
                    break;
                case 4:
                    a.F =  50.;
                    break;

                default:
                    cout << "Bad action" << endl;
                    exit(1);
                    break;
            }
        }
        
        for (int i=0; i<nssteps; i++)
        {
            for (auto& a : agents)
                a.u = rk46_nl(t, dt, a.u, bind(&CartPole::D, &a, placeholders::_1, placeholders::_2));
            
            t += dt;
        }
        
        printf("Sending state! \n");
        for (auto& a : agents)
        {
            double r = 0.0;//1. - fabs(a.u.y3);
            bool kill = false;
            ntot += 1; sincelast += 1;
            
            if ( (fabs(a.u.y3) > 0.2) || (fabs(a.u.y1) > 2.4) )
            {
                r = -1; nfallen += 1;
                percfallen = nfallen/ntot;
                sincelast = 0; kill = true;
            }
            
            state[0] = a.u.y1;
            state[1] = a.u.y2;
            state[2] = a.u.y4;
            state[3] = a.u.y3;
            comm->sendState(state, r);
            
            if(kill)
            {   //begin anew
                a.u = Vec4( .02*(drand48()-.5), .02*(drand48()-.5), .02*(drand48()-.5), .02*(drand48() -.5));
                a.F = 0;
                first = true;
            }
        }
        
        if (ntot % 10000 == 0)
        {
            cout << nfallen - duringlast << endl;
            duringlast =+ nfallen;
        }
    }
    
    return 0;
}
