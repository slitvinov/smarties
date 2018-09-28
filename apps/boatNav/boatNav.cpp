#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <array>
#include <functional>
#include "Communicator.h"
#include "odeSolve.h"

#define dt 1.0e-4
#define SAVEFREQ 1
#define STEPFREQ 1
#define maxStep = 500000

using namespace std;

struct Entity
{
	const modelParams params;
	const unsigned nStates;

	double pathStart[2], pathEnd[2];
	double thetaPath;

	std::mt19937& genA;
	bool isOver=false;

	vector<double> u,v,r, uDot, vDot, rDot, x,y,thetaR;
	u.reserve(maxStep), v.reserve(maxStep), r.reserve(maxStep);
	uDot.reserve(maxStep), vDot.reserve(maxStep), rDot.reserve(maxStep);
	x.reserve(maxStep), y.reserve(maxStep), thetaR.reserve(maxStep);

	Entity(std::mt19937& _gen, const unsigned nQ, const double xS[2], const double xE[2]) : nStates(nQ), genA(_gen)
	{
		pathStart[0] = xS[0]; 
		pathStart[1] = xS[1];
		pathEnd[0] = xE[0]; 
		pathEnd[1] = xE[1];
		thetaPath = atan2( (pathEnd[1] - pathStart[1]), (pathEnd[0] - pathStart[0]) ); 
	}

	array<double, 2> p; // absolute position
	double rPos, thetaPos; // wrt starting position
	double thetaNose; // Which way is the boat pointing (wrt path) 

	void reset() {

		normal_distribution<double> distribX(0, 2*params.l);
		normal_distribution<double> distribAng(0, M_PI/3.0); // allow random deviation by 60 degrees

		p[0] = this->pathStart[0] + distribX(genA);
		p[1] = this->pathStart[1] + distribX(genA);

		thetaNose = this->thetaPath + distribAng(genA);
		isOver = false;

		u.push_back(0.0), v.push_back(0.0), r.push_back(0.0);
		uDot.push_back(0.0), vDot.push_back(0.0), rDot.push_back(0.0);
		x.push_back(p[0]), y.push_back(p[1]), thetaR.push_back(thetaNose - thetaPath);
	}

	bool is_over() {
		return isOver; 
	}

	// actions are thrustLeft and thrustRight
	void advance(vector<double> act) {
		assert(act.size() == 2);

		// Assume no other forces for the time being
		const double thrustL = act[0];
		const double thrustR = act[1];
		const double torque = 0.5*params.l*(thrustR - thrustL);

		const double uN[3] = {u.back(), v.back(), r.back()};
		const double uDotN[3] = {uDot.back(), vDot.back(), rDot.back()};
		double uNp1[3] = {0.0};
		double uDotNp1[3] = {0.0};

		odeSolve(uN, uDotN, params, dt, thrustL, thrustR, torque, uNp1, uDotNp1);
		u.push_back(uNp1[0]), v.push_back(uNp1[1]), r.push_back(uNp1[2]);
		uDot.push_back(uDotNp1[0]), vDot.push_back(uDotNp1[1]), rDot.push_back(uDotNp1[2]);

		const double xN[3] = {x.back(), y.back(), thetaR.back()};
		double xNp1[3] = {0.0};
		trajectory(xN, uN, uNp1, dt, xNp1);
		x.push_back(xNp1[0]), y.push_back(xNp1[1]), thetaR.push_back(xNp1[2]);

	}

	double getAngle() const {
		const double relX = p[0] - pathStart[0];
		const double relY = p[1] - pathStart[1];
		return std::atan2(relY, relX) - thetaPath;
	}

	double getDistance() const {
		double relX = E.p[0] - p[0];
		double relY = E.p[1] - p[1];
		return std::sqrt(relX*relX + relY*relY);
	}
};


struct USV: public Entity
{
  const double stdNoise; // Only boat assumed to suffer from noise

  USV(std::mt19937& _gen, const unsigned _nStates, const double vM, const double dN)
    : Entity(_gen, _nStates, vM), stdNoise(dN) {}

  // State vector contains : 
  template<typename T>
  vector<double> getState() { // wrt path

	  vector<double> state(nStates, 0);

	  state[0] = p[0]; // dx
	  state[1] = p[1]; // dy
	  const double angEnemy1 = getAngle(E1);
	  const double distEnemy = distEnemy1*distEnemy2;

	  const double ETA = distEnemy/fabs(speed);
	  const double noiseAmp = stdNoise*ETA;
	  std::normal_distribution<double> distrib(0, noiseAmp); // mean=0, stdDev=noiseAmp

	  const double noisyDy = angEnemy1 + distrib(genA);
	  state[2] = distEnemy1*std::cos(noisyAng1); 
	  return state;
  }

  double getReward(const ) const {
	  return -/params.l;
  }

  vector<bool> checkTermination() {
	  const double threshold= 0.1*this->width;
	  const double distGoal = getDistance(E1);
	  const bool goal = (distCheckpoint < threshold) ? true : false;
	  if (goal) isOver = true; 
	  return goal;
  }
 
};


int main(int argc, const char * argv[])
{

  //const unsigned maxStep = 500000;
  const int control_vars = 2; // 2 components of thrust
  const int state_vars = 6;   // number of states (self pos[2], )
  const int number_of_agents = 1;

  //communication:
  const int socket = std::stoi(argv[1]);
  //socket number is given by RL as first argument of execution
  Communicator 	comm(socket, state_vars, control_vars, number_of_agents);
  SUV 		boat(comm.gen, state_vars, 0.0);

  unsigned sim = 0;
  while(true) //train loop
  {
    //reset environment:
    boat.reset();

    //send initial state
    comm.sendInitState(boat.getState(), 0);

    unsigned step = 0;
    while (true) //simulation loop
    {
      boat.advance(comm.recvAction(0), step);

	  vector<bool> reachedGoal = boat.checkTermination();
	  if(boat.is_over()){ // Terminate simulation 
		  const double finalReward = 10*boat.params.l;
		  comm.sendTermState(boat.getState(),finalReward, 0);

		  printf("Sim #%d reporting that boat reached safe harbor.\n", sim); fflush(NULL);
		  sim++; break;
	  }

      if(step++ < maxStep)
      {
        comm.sendState(boat.getState(), boat.getReward(), 0);
      }
      else
      {
        comm.truncateSeq(boat.getState(), boat.getReward(), 0);
        sim++;
        break;
      }
    }
  }
  return 0;
}
