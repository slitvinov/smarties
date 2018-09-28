#include "odeSolve.h"

void getDerivs(const modelParams p, const double Fx, const double Fy, const double Tau, const double u, const double v, const double r, const double vDot, const double rDot, double retVal[3])
{
	retVal[0] = (Fx + r*(p.YvDot*v + p.m*v - r*(p.NvDot + p.YrDot)/2) - u*(p.Xu + p.Xuu*u)) / (p.m-p.XuDot);
	retVal[1] = (Fy - r*(p.XuDot*u + p.m*u) - p.Yv*v + p.YrDot*rDot - p.Yr*r) / (p.m-p.YvDot);
	retVal[2] = (Tau + p.NvDot*vDot - p.Nv*v - u*(p.YvDot*v + p.m*v + r*(p.NvDot+p.YrDot)/2) + p.Nr*r + v*(p.XuDot*u + p.m*u)) / (p.Izz-p.NrDot);
}


//////////////////////////////////////////////////////////////////////////////////////////
void odeSolve(const double uVec[3], const double uDotVec[3], const modelParams params, const double dt, 
		const double fX, const double fY, const double torque,
		double uOut[3], double uDotOut[3]) 
{
	const double u=uVec[0], v=uVec[1], r=uVec[2];
	const double uDot=uDotVec[0], vDot=uDotVec[1], rDot=uDotVec[2];

	//RK4
	double stage1[3]={0}, stage2[3]={0}, stage3[3]={0}, stage4[3]={0};
	getDerivs(params, fX, fY, torque, u, v, r, vDot, rDot, stage1);
	getDerivs(params, fX, fY, torque, u+(0.5*dt*stage1[0]), v+(0.5*dt*stage1[1]), r+(0.5*dt*stage1[2]), vDot, rDot, stage2);
	getDerivs(params, fX, fY, torque, u+(0.5*dt*stage2[0]), v+(0.5*dt*stage2[1]), r+(0.5*dt*stage2[2]), vDot, rDot, stage3);
	getDerivs(params, fX, fY, torque, u+(0.5*dt*stage3[0]), v+(0.5*dt*stage3[1]), r+(0.5*dt*stage3[2]), vDot, rDot, stage4);

	uDotOut[0] = (stage1[0] + 2*stage2[0] + 2*stage3[0] + stage4[0]);
	uDotOut[1] = (stage1[1] + 2*stage2[1] + 2*stage3[1] + stage4[1]);
	uDotOut[2] = (stage1[2] + 2*stage2[2] + 2*stage3[2] + stage4[2]);

	uOut[0] = u + (dt/6.0)*uDotOut[0];
	uOut[1] = v + (dt/6.0)*uDotOut[1];
	uOut[2] = r + (dt/6.0)*uDotOut[2];
}

//////////////////////////////////////////////////////////////////////////////////////////
void trajectory(const double xVec[3], const double uN[3], const double uNp1[3], const double dt, double xOut[3]) 
{
	xOut[0] = xVec[0] + (dt/2.0)*(uN[0] + uNp1[0]);
	xOut[1] = xVec[1] + (dt/2.0)*(uN[1] + uNp1[1]);
	xOut[2] = xVec[2] + (dt/2.0)*(uN[2] + uNp1[2]);
}
