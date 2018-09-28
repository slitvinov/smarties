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

void odeSolve(const double uVec[3], const double uDotVec[3], const modelParams params, const double dt, const double fX, const double fY, const double torque, double uOut[3], double uDotOut[3]);

void trajectory(const double xVec[3], const double uN[3], const double uNp1[3], const double dt, double xOut[3]);
