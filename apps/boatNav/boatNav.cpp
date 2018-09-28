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

using namespace std;

struct Entity
{
	const modelParams params;
	const unsigned nStates;
	const double maxSpeed = 5*params.l;

	double pathStart[2], pathEnd[2];
	double thetaPath;

	std::mt19937& genA;
	bool isOver=false;

	Entity(std::mt19937& _gen, const unsigned nQ, const double xS[2], const double xE[2]) : nStates(nQ), genA(_gen)
	{
		pathStart[0] = xS[0]; 
		pathStart[1] = xS[1];
		pathEnd[0] = xE[0]; 
		pathEnd[1] = xE[1];
		thetaPath = atan2( (pathEnd[1] - pathStart[1]), (pathEnd[0] - pathStart[0]) ); 
	}

	array<double, 2> p; // absolute position
	double deltaX, deltaY, deltaTheta; // wrt straight line track
	double theta; 

	void reset() {

		normal_distribution<double> distribX(0, 2*params.l);
		normal_distribution<double> distribAng(0, M_PI/3.0); // allow deviation by 60 degrees

		p[0] = this->pathStart[0] + distribX(genA);
		p[1] = this->pathStart[1] + distribX(genA);

		theta = this->thetaPath + distribAng(genA);
		isOver = false;
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

		odeSolve(uNot, params, N, tt, forceX, forceY, torque, u, v, r);
		trajectory(N, tt, u, v, r, p, theta);

	}

	template<typename T>
		unsigned getAngle(const T& E) const {
			const double relX = p[0] - pathStart[0];
			const double relY = p[1] - pathStart[1];
			return std::atan2(relY, relX) + theta - thetaPath;
		}

	template<typename T>
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

  template<typename T>
  double getReward(const ) const {
	  return -/params.l;
  }

  template<typename T>
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

  const unsigned maxStep = 500000;
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
      boat.advance(comm.recvAction(0));

	  vector<bool> reachedGoal = boat.checkTermination();
	  if(boat.is_over()){ // Terminate simulation 
		  const double finalReward = 10*boat.params.l;
		  comm.sendTermState(boat.getState(),finalReward, 0);

		  printf("Sim #%d reporting that boat reached safe harbor.\n", sim); fflush(NULL);
		  sim++; break;
	  }

      if(step++ < maxStep)
      {
        comm.sendState(  boat.getState(), boat.getReward(), 0);
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
