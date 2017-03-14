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
    const double I1 = m1*l1/12.;    //moment of inertia of arms
    const double I2 = m2*l2/12.;    //moment of intertia of legs
    const double g = 9.81;  //gravity
    int info;
    Vec4 u;
    double F; //torque at the second joint
    double t, time_since_up;

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

double mapTheta1_to_2pi(double theta)
{
//theta between -2pi and 2pi
 theta = fmod(theta,2*M_PI);
 if(theta<0) theta += 2*M_PI;
//theta between 0 and 2*pi
 return theta;
}

Communicator * comm;
int main(int argc, const char * argv[])
{
    const int n = 1; //n agents
    //communication:
    const int sock = std::stoi(argv[1]);
    //time stepping
    const double dt = 1e-3;
    double t = 0;
    std::mt19937 gen(sock);
    std::uniform_real_distribution<double> distribution(-.2,.2);
    //communicator class, it needs a socket number sock, given by RL as first argument of execution
    Communicator comm(sock,4,1);
    //vector of state variables: in this case theta1, ang_velocity1, theta2, ang_velocity2
    vector<double> state(4);
    //vector of actions received by RL
    vector<double> actions(1);

    //random initial conditions:
    vector<Acrobot> agents(n);
    for (auto& a : agents) {
        a.u = Vec4(distribution(gen)+M_PI, distribution(gen), distribution(gen), distribution(gen));
        a.u.y1 = mapTheta1_to_2pi(a.u.y1);
        a.F    = 0; a.time_since_up = 0; 
        a.info = 1;
    }
    
    while (true) {
        int k(0); //agent ID, for now == 0
        for (auto& a : agents) { //assume we have only one agent per application for now...
            double r = 0.0;
           
            //load state:
            //theta1 is between 0 and 2pi, 0 and 2pi are vertical down
            state[0] = a.u.y1;
            state[1] = a.u.y2;
            state[2] = a.u.y4;
            state[3] = a.u.y3;
            
            r = -pow(cos(a.u.y1),3);
		
            //printf("Sending state %f %f %f %f\n",state[0],state[1],state[2],state[3]); fflush(0);
            //printf("Current reward %f\n", r); fflush(0);
            ///////////////////////////////////////////////////////
            // arguments of comm->sendState(k, a.info, state, r)

            //k is agent id, if only one agent in the game: k=0

        	//info is: 1 for the initial state communicated to the RL
            //  	   0 for any following communication, except
        	//		   2 for the terminal state (meaning NO ACTION REQUIRED)
            ///////////////////////////////////////////////////////
            comm.sendState(k, a.info, state, r);
            comm.recvAction(actions);

            //printf("Acrobot acting %f from state %f %f %f %f\n", actions[0],state[0],state[1],state[2],state[3]); fflush(0);

            a.F = actions[0];
            a.info = 0; //at least one comm is done, so i set info to 0

            //printf("Received action %f\n", a.F); fflush(0);

        	//advance the sim:
            for (int i=0; i<50; i++) {
#if 1
                if ( a.u.y3 > 0.95*M_PI ) {   //legs in an inhumane position
                    //acrobot is not capable of executing any torque in this position
                    a.u.y3=0.94999*M_PI; //legs too far in the front
                    a.u.y4=0.;
                } else if (a.u.y3 < -.25*M_PI) {
                    a.u.y3=-.25001*M_PI; //legs too far in the back
                    a.u.y4=0.;
                }
#endif
                a.u = rk46_nl(a.t, dt, a.u, bind(&Acrobot::D, &a, placeholders::_1, placeholders::_2));
                    
                if (a.u.y3 > 2*M_PI || a.u.y3 < -2*M_PI) abort(); //should never happen
                    
                a.u.y1 = mapTheta1_to_2pi(a.u.y1);
                    
                a.t += dt;
                
                //added a variable: how much since acrobot was
                // - more or less up ( a.u.y1 \approx M_PI )
                // - with the legs stretched ( a.u.y3 \approx 0 )
                // - relatively slow
                if(a.u.y1 > .75*M_PI && a.u.y1 < 1.25*M_PI && fabs(a.u.y3) < .25*M_PI && fabs(a.u.y2) < 1 )
                    a.time_since_up = a.t;
                
                //check if terminal state has been reached:
                bool failed = fabs(a.u.y2)>10*M_PI || fabs(a.u.y4)>10*M_PI ||  a.t-a.time_since_up > 10;
                if (failed)
                {
                    a.info = 2; //tell RL we are in terminal state
                    double r = -10.; //give terminal reward
                    state[0] = a.u.y1;
                    state[1] = a.u.y2;
                    state[2] = a.u.y4;
                    state[3] = a.u.y3;
                //    printf("Sending term state %f %f %f %f\n",state[0],state[1],state[2],state[3]); fflush(0);
                    comm.sendState(k, a.info, state, r);

                    //re-initialize the simulations (random initial conditions):
                    a.u = Vec4(distribution(gen)+M_PI, distribution(gen), distribution(gen), distribution(gen));
                    a.u.y1 = mapTheta1_to_2pi(a.u.y1);
                    a.t = 0.; a.time_since_up = 0.;
                    a.F = 0.;
                    a.info = 1; //set info back to initial state
                    r = 0.;
                    break;
                }
        }
    }
}
    
    return 0;
}
