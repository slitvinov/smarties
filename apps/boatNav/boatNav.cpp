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
#define maxStep 100000

using namespace std;

vector<double> u,v,r, uDot, vDot, rDot, x,y,thetaR;
vector<double> tt, forceX;

struct Entity
{
	const modelParams params;
	const unsigned nStates;

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

		u.reserve(maxStep); v.reserve(maxStep); r.reserve(maxStep);
		uDot.reserve(maxStep); vDot.reserve(maxStep); rDot.reserve(maxStep);
		x.reserve(maxStep); y.reserve(maxStep); thetaR.reserve(maxStep);
		tt.reserve(maxStep); forceX.reserve(maxStep);
	}

	array<double, 2> p; // absolute position
	double rPos, thetaPos; // wrt starting position
	double thetaNose; // Which way is the boat pointing (wrt path) 

	void reset() {

		normal_distribution<double> distribX(0, params.l);
		normal_distribution<double> distribU(0, 0.5*params.l);
		normal_distribution<double> distribAng(0, M_PI/18.0); // allow random deviation by 10 degrees

		p[0] = this->pathStart[0] + distribX(genA); // random starting positions
		p[1] = this->pathStart[1] + distribX(genA);


		thetaNose = this->thetaPath + distribAng(genA);
		isOver = false;

		u.push_back(distribU(genA)), v.push_back(distribU(genA)); // random initial linear velocities
		r.push_back(0.0);
		uDot.push_back(0.0), vDot.push_back(0.0), rDot.push_back(0.0);
		x.push_back(p[0]), y.push_back(p[1]), thetaR.push_back(thetaNose - thetaPath);
		tt.push_back(0.0);
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

		p[0] = xNp1[0];
		p[1] = xNp1[1];

	}

	double getAngle(const double xLoc[2]) const {
		const double relX = p[0] - xLoc[0];
		const double relY = p[1] - xLoc[1];
		return std::atan2(relY, relX) - thetaPath;
	}

	double getDistance(const double xLoc[2]) const {
		double relX = p[0] - xLoc[0];
		double relY = p[1] - xLoc[1];
		return std::sqrt(relX*relX + relY*relY);
	}
};


struct USV: public Entity
{
  const double stdNoise; // Only boat assumed to suffer from noise

  USV(std::mt19937& _gen, const unsigned _nStates, const double dN, const double xPathStart[2], const double xPathEnd[2])
    : Entity(_gen, _nStates, xPathStart, xPathEnd), stdNoise(dN) {}

  // State vector contains : 
  vector<double> getState() const { // wrt path

	  vector<double> state(nStates, 0);

	  state[0] = getDistance(pathStart); 	// rPos
	  state[1] = getAngle(pathStart); 	// thetaPos
	  state[2] = thetaR.back(); 		// bearing nose
	  state[3] = u.back();
	  state[4] = v.back();
	  state[5] = r.back();

	  //const double noiseAmp = stdNoise*ETA;
	  //std::normal_distribution<double> distrib(0, noiseAmp); // mean=0, stdDev=noiseAmp
	  //const double noisyDy = angEnemy1 + distrib(genA);
	  //state[2] = distEnemy1*std::cos(noisyAng1); 
	  return state;
  }

  double getReward() const {
	  // Compute perpendicular distance from path
	  const double thetaStart = getAngle(pathStart);
	  const double thetaEnd = getAngle(pathEnd);

	  const double distStart = getDistance(pathStart);
	  const double distEnd 	= getDistance(pathEnd); 

	  double retVal;
	  if( abs(thetaEnd) >= M_PI/2.0 && (thetaStart>=-M_PI/2.0 && thetaStart<=M_PI/2.0) )
	  {
		  retVal = distStart*abs(sin(thetaStart));
		  //printf("getReward sez %f, %f, %f\n", p[0], p[1], retVal);
	  } else {
		  retVal = (distStart < distEnd) ? distStart : distEnd;
		  //printf("outside line bounds, returning distance from nearest harbour %f\n", retVal);
	  }

	  return -retVal/params.l; // normalize with ship width
  }

  bool checkTermination() {
	  const double threshold= 0.1*this->params.l; // within 10% of ship width
	  const double distGoal = getDistance(pathEnd);
	  const bool goal = (distGoal < threshold) ? true : false;
	  if (goal) isOver = true; 

	  // Check if need to abort
	  const double latDist = -getReward(); // already normalized by params.l
	  const bool abortSim = (latDist > 10) ? true : false ;
	  if (abortSim) {
		  printf("boat is too far from path, distance = %f boatWidths\n", latDist);
		  isOver = true;
	  }

	  return abortSim;
  }
 
};


int main(int argc, const char * argv[])
{

  const int control_vars = 2; // 2 components of thrust
  const int state_vars = 6;   // number of states
  const int number_of_agents = 1;
  char fileName[100];

  //communication:
  const int socket = std::stoi(argv[1]);
  //socket number is given by RL as first argument of execution
  Communicator	comm(socket, state_vars, control_vars, number_of_agents);
  std::mt19937	&rngPointer =  comm.getPRNG();

  // Path start and end
  const double xPathStart[2] = {0,0};
  const double xPathEnd[2] = {10,0};

  USV	boat(rngPointer, state_vars, 0.0, xPathStart, xPathEnd);

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
	    vector<double> actions(control_vars, 0.0);
	    actions = comm.recvAction(0);

// Overwrite actions
const double angFactor = 2*M_PI/10.0;
actions[0] = cos(tt[step]*angFactor/4.0); // Thruster Left
actions[1] = 0.0; // Thruster Right

	    forceX[step] = actions[0] + actions[1];
	    boat.advance(actions);
	    tt.push_back(dt*(step+1));

	  const bool abortSim = boat.checkTermination();// Abort if too far from trajectory
	  if(boat.is_over()){ // Terminate simulation 
			  
		  const double finalReward = 10*boat.params.l;

		  if(abortSim){
			  comm.sendTermState(boat.getState(), -finalReward, 0);
			  printf("Sim #%d reporting that boat got lost.\n", sim); fflush(NULL);
		  } else {
			  comm.sendTermState(boat.getState(), +finalReward, 0);
			  printf("Sim #%d reporting that boat reached safe harbor.\n", sim); fflush(NULL);
		  }
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

    sprintf(fileName, "learnedTrajectory_%07d.txt", sim);
    FILE *temp = fopen(fileName, "w");
    fprintf(temp, "time \t forceX \t forceY \t u \t v \t r \t x \t y \t theta\n");
    for(int i=0; i<tt.size(); i++)  fprintf(temp, "%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n", tt[i], forceX[i], 0.0, u[i], v[i], r[i], x[i], y[i], thetaR[i]);
    fclose(temp);

  }
  return 0;
}
