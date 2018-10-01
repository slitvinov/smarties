#pragma once

struct modelParams
{
        double m = 280;
        double Izz = 300;
        double l = 1.83;
        double Xu = 86.45;
        double Xuu = 0;
        double Yv = 300;
        double Nr = 500;
        double Nv = -250;
        double Yr = -80;
        double Nu = 20;
        double XuDot = -30;
        double YvDot = -40;
        double NrDot = -90;
        double NvDot = -50;
        double YrDot = -50;
};

void odeSolve(const double uNot[3], const modelParams params, const int N, const double *tt, const double *fX, const double *fY, const double *torque, double *u, double *v, double *r);

void trajectory(const int N, const double *tt, const double *u, const double *v, const double *r, double *x, double *y, double *theta);
