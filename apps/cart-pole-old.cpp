//
//  main.cpp
//  cart-pole
//
//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
//

#include                               <iostream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <functional>

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
    const double m = 0.1;
    const double M = 1;
    const double l = 0.5;
    const double g = 9.81;
    
    Vec4 u;
    double F;
    
    Vec4 D(Vec4 u, double t)
    {
        Vec4 res;
        
        const double cosy = cos(u.y3);
        const double siny = sin(u.y3);
        const double tmp1 = 1. / (M+m*(1-cosy));
        const double tmp2 = u.y4*u.y4*l;
        
        const double fac1 = 1./(M+m);
        const double fac2 = l*(4./3. - fac1*(m*cosy*cosy));
        
        const double F1 = F + m*l*u.y4*u.y4*siny;
        
        res.y4 = (g*siny - fac1*F1*cosy)/fac2;
        res.y2 = fac1*(F1 - m*l*res.y4*cosy);
        //printf("Angular velocity due to gravity %f and forcing %f (F=%f)\n", g*siny, -fac1*F1*cosy, F);
        res.y1 = u.y2;
        //res.y2 = m * tmp1 * ((tmp2 - cosy*g)*siny + F/m);
        res.y3 = u.y4;
        //res.y4 = tmp1/l * (siny * (g*(m+M) - m*tmp2) - F);
        return res;
    }
};


int main(int argc, const char * argv[])
{
    const int n = 1;
    long long int nfallen = 0;
    long long int sincelast = 0;
    long long int duringlast = 0;
    long long int ntot = 0;
    double percfallen = 0.0;
    const int nssteps = 1;
    
    const double dt = 1e-2;
    double t = 0;
    bool first = true;
    vector<CartPole> agents(n);
    for (auto& a : agents)
    {
        a.u = Vec4( .02*(drand48()-.5), .02*(drand48()-.5), .02*(drand48()-.5), .02*(drand48() -.5));
        a.F    = 0;
    }

    string keyword;
    
    while (ntot<=1250000)
    {
        if (first)
        {
            cerr << "States and rewards:" << endl;
            for (auto& a : agents)
            {
                double r = 0.0;
                cerr << a.u.y1 << " " << a.u.y2 << " " << a.u.y4 << " " << a.u.y3 << " ";
                cerr << r << endl;
                cout << a.u.y1 << " " << a.u.y2 << " " << a.u.y4 << " " << a.u.y3 << " ";
                cout << r << endl;
            }
            first = false;
        }
        
        cin >> keyword;
        if (keyword == "Actions:")
        {
            int act;
            for (auto& a : agents)
            {
                cin >> act;
                
                switch (act)
                {
                    case 0:
                        a.F = -20.;
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
                        a.F =  20.;
                        break;

                    default:
                        cout << "Bad action" << endl;
                        exit(1);
                        break;
                }
            }
        }
        else
        {
            cout << "Bad keyword '" << keyword << "'" << endl;
            exit(2);
        }
        
        for (int i=0; i<nssteps; i++)
        {
            for (auto& a : agents)
            {
                a.u = rk46_nl(t, dt, a.u, bind(&CartPole::D, &a, placeholders::_1, placeholders::_2));
            }
            
            t += dt;
        }
        
        cerr << "States and rewards:" << endl;
        for (auto& a : agents)
        {
            double r = 0;
            bool kill = false;
            //if (fabs(a.u.y1) > 2)        r -= 1;
            //if (fabs(a.u.y3) > 1)        r -= 10;
            ntot += 1;
            sincelast += 1;
            if ( (fabs(a.u.y3) > 1)) //(fabs(a.u.y1) > 2) ||
            {
                
                r = -1;
                nfallen += 1;
                percfallen = nfallen/ntot;
                //cout << "An other one bites the dust. " << sincelast << " actions ago I last fell. Risk at: " << percfallen << " %." << endl;
                //cout << sincelast << endl;
                sincelast = 0;
                kill = true;
            }
            
            cerr << a.u.y1 << " " << a.u.y2 << " " << a.u.y4 << " " << a.u.y3 << " ";
            cerr << r << endl;
            
            if(kill)
            {   //begin anew
                a.u = Vec4( .02*(drand48()-.5), .02*(drand48()-.5), .02*(drand48()-.5), .02*(drand48() -.5));
                a.F    = 0;
                first = true;
            }
        }
        if (ntot % 10000 == 0) {
            cout << nfallen - duringlast << endl;
            duringlast =+ nfallen;
        }
    }
    
    return 0;
}
