#include <iostream>
#include <cmath>
#include <cstdio>
#include <vector>
#include <functional>
#define _SMART_
//#define _PRINT_
using namespace std;

struct Vec6
{
    double u,v,w,x,y,a; //x vel, y vel, ang vel, x pos, y pos, angle (or their time derivatives)
    
    double vx()
    {
        const double sina = sin(a);
        const double cosa = cos(a);
        return u*cosa + v*sina;
    }
    
    double vy()
    {
        const double sina = sin(a);
        const double cosa = cos(a);
        return v*cosa - u*sina;
    }
    
    Vec6(double u=0, double v=0, double w=0, double x=0, double y=0, double a=0) : u(u), v(v), w(w), x(x), y(y), a(a) {};
    
    Vec6 operator*(double s) const
    {
        return Vec6(u*s, v*s, w*s, x*s, y*s, a*s);
    }
    
    Vec6 operator+(const Vec6& s) const
    {
        return Vec6(u+s.u, v+s.v, w+s.w, x+s.x, y+s.y, a+s.a);
    }
};

/* 
 Julien Berland, Christophe Bogey, Christophe Bailly,
 Low-dissipation and low-dispersion fourth-order Runge-Kutta algorithm,
 Computers & Fluids, Volume 35, Issue 10, December 2006, Pages 1459-1463, ISSN 0045-7930,
 http://dx.doi.org/10.1016/j.compfluid.2005.04.003
 */

template <typename Func, typename Vec>
Vec6 rk46_nl(double t0, double dt, Vec u0, Func&& Diff)
{
    const double a[] = {0.000000000000, -0.737101392796, -1.634740794341, -0.744739003780, -1.469897351522, -2.813971388035};
    const double b[] = {0.032918605146,  0.823256998200,  0.381530948900,  0.200092213184,  1.718581042715,  0.270000000000};
    const double c[] = {0.000000000000,  0.032918605146,  0.249351723343,  0.466911705055,  0.582030414044,  0.847252983783};
    
    const int s = 6;
    Vec6 w;
    Vec6 u(u0);
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
    const double II  = 20.0;
    const double beta= 0.1;
    const double piinv = 1/3.14159265359;
    const double CR = 3.14159265359;
    
    Vec6 s;
    double T, dT; //torque, angular jerk
    
    Vec6 D(Vec6 s, const double t)
    {
        Vec6 res;
        
        const double uv2p = s.u*s.u + s.v*s.v;
        const double suv2 = sqrt(uv2p);
        const double uv2n = s.u*s.u - s.v*s.v;
        
        const double _f1  = suv2==0 ? 0.0 : s.u*s.v/suv2;
        const double _f2  = uv2p==0 ? 0.0 : uv2n/uv2p;
        const double Gamma = (2*piinv)*(-CT*_f1 + CR*s.w);
        const double F = piinv*(Aa - Bb*_f2)*suv2*s.u;
        const double G = piinv*(Aa - Bb*_f2)*suv2*s.v;
        const double M = (mut + nut*fabs(s.w))*s.w;
        
        const double sinth = sin(s.a);
        const double costh = cos(s.a);
        
        const double fact1 = II + beta*beta;
        const double fact2 = II + 1.;
        const double fact3 = 0.25*(II*(1.0+beta*beta) + 0.5*(1-beta*beta)*(1-beta*beta));
        
        res.u = ( fact2*s.v*s.w - Gamma*s.v - sinth - F)/fact1;
        res.v = (-fact1*s.u*s.w + Gamma*s.u - costh - G)/fact2;
        res.w = ((beta*beta-1.0)*s.u*s.v + T - M)/fact3;
        res.x = s.u*costh - s.v*sinth;
        res.y = s.u*sinth + s.v*costh;
        res.a = s.w;

        return res;
    }
    
    void print(const double t, const int ID)
    {
        string suff = "trajectory_" + to_string(ID) + ".dat";
        FILE * f = fopen(suff.c_str(), t==0. ? "w" : "a");
        fprintf(f,"%e %e %e %e %e %e %e %e\n",t,s.u,s.v,s.w,s.x,s.y,s.a,T);
        fclose(f);
    }
};

int main(int argc, const char * argv[])
{
    const int n = 1; //THIS CODE IS WRONG FOR >1 AGENTS! (TODO)
    const int nssteps = 10;
    const double th_f  = 3.14159265359/4.;
    const double dt = 5e-3;
    const double dT = 1.;
    double t = 0;
    bool first = true;
    vector<Glider> agents(n);
    for (auto& a : agents)
    {
        a.s.u = 0.0;//0.2 * (drand48() - 0.5);
        a.s.v = 0.0;//0.2 * (drand48() - 0.5);
        a.s.w = 0.0;//0.2 * (drand48() - 0.5);
        a.s.x = -100.;//0.2 * (drand48() - 0.5);
        a.s.y = 50.0;//0.2 * (drand48() - 0.5);
        a.s.a = 0.0;//0.2 * (drand48() - 0.5);
        a.T   = 0.0;
    }
    
#ifdef _SMART_
    cerr << n << " agents" << endl;
    string keyword;
#endif
    
    while (t<5000000000000)
    {
#ifdef _SMART_
        // if starting: take a dummy action (do nothing) and communicate initial state to smarties (smarties always needs an old state and a new state)
        if (first)
        {
            cin >> keyword;
            if (keyword == "Actions:")
            {
                int act;
                for (auto& a : agents)
                    cin >> act;
            }
            else
            {
                cout << "Bad keyword '" << keyword << "'" << endl;
                exit(2);
            }
            cerr << "States and rewards:" << endl;
            for (auto& a : agents)
            {
                double r = 0.0;
                cerr <<a.s.vx()<<" "<<a.s.vy()<<" "<<a.s.w<<" "<<a.s.x<<" "<<a.s.y<<" "<<a.s.a<<" "<<a.T<<" ";
                cerr << r << endl;
            }
            first = false;
        }
        
        // take proper action
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
                        a.dT =-dT; //negative angular jerk
                        break;
                    case 1:
                        a.dT = 0.;
                        break;
                    case 2:
                        a.dT = dT; //positive angular jerk
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
        
        //perform a number of time steps with that action
        for (int i=0; i<nssteps; i++)
        {
#endif
            int id = 0;
            for (auto& a : agents)
            {
                id++;
                a.T += a.dT*dt;
                a.s = rk46_nl(t, dt, a.s, bind(&Glider::D, &a, placeholders::_1, placeholders::_2));
#ifdef _PRINT_
                a.print(t,id);
#endif
            }
            
            t += dt;
#ifdef _SMART_
        }
        
        //communicate state to smarties
        cerr << "States and rewards:" << endl;
        for (auto& a : agents)
        {
            double r = 0.0;
            bool kill = false;
            if (a.s.y <= 0.) //ground reached
            {
                a.s.y = 0.; // avoids problems in smarties
                r += 1 - (a.s.x*a.s.x/1e4 + (a.s.a-th_f)*(a.s.a-th_f)/(th_f*th_f)); //goal: x = 0.0, theta = pi/4
                kill = true;
            }
            else if (fabs(a.T)>1.) //limit on the maximum torque, same as Paoletti's paper
            {
                r -= 1000;
                kill = true;
            }
            
            cerr <<a.s.vx()<<" "<<a.s.vy()<<" "<<a.s.w<<" "<<a.s.x<<" "<<a.s.y<<" "<<a.s.a<<" "<<a.T<<" ";
            cerr << r << endl;
            
            if(kill)
            {   //begin anew
#ifdef _PRINT_
                exit(2); //just want to plot one trajectory!
#endif
                a.s = Vec6(0.0,0.0,0.0,-100.,50.,0.0);
                a.T = 0.;
                first = true;
            }
        }
#endif
    }
    
    return 0;
}
