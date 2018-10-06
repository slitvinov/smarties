#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <array>
#include <functional>
#include "Communicator.h"
#include "odeSolve.h"

#define dt 1.0e-2
#define maxStep 20000

using namespace std;

vector<double> u,v,r, x,y,thetaR;
vector<double> tt, forceX, thrustL, thrustR, torque;
const double forceY = 0.0;

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
		x.reserve(maxStep); y.reserve(maxStep); thetaR.reserve(maxStep);
		tt.reserve(maxStep); forceX.reserve(maxStep); thrustL.reserve(maxStep); thrustR.reserve(maxStep); torque.reserve(maxStep);
	}

	array<double, 2> p; // absolute position
	double rPos, thetaPos; // wrt starting position
	double thetaNose; // Which way is the boat pointing (wrt path) 

	void reset() {

		// Reset all global vectors - won't destroy the capacity
		u.clear(); v.clear(); r.clear();
		x.clear(); y.clear(); thetaR.clear();
		tt.clear(); forceX.clear(); thrustL.clear(); thrustR.clear(); torque.clear();

		normal_distribution<double> distribX(0, params.l);
		normal_distribution<double> distribU(0, 2*params.l);
		normal_distribution<double> distribAng(0, M_PI/18.0); // allow random deviation by 10 degrees

		p[0] = this->pathStart[0] + distribX(genA); // random starting positions
		p[1] = this->pathStart[1] + distribX(genA);

		thetaNose = this->thetaPath + distribAng(genA);
		isOver = false;

		u.push_back(distribU(genA)), v.push_back(distribU(genA)); // random initial linear velocities
		r.push_back(0.0);
		x.push_back(p[0]), y.push_back(p[1]), thetaR.push_back(thetaNose - thetaPath);
		tt.push_back(0.0);
	}

	bool is_over() {
		return isOver; 
	}

	// actions are thrustLeft and thrustRight
	void advance() {

		// Assume no other forces for the time being
		torque.push_back(0.5*params.l*(thrustR.back() - thrustL.back()));

		const double uN[3] = {u.back(), v.back(), r.back()};
		double uNp1[3] = {0.0};

		odeSolve(uN, params, dt, forceX.back(), forceY, torque.back(), uNp1);
		u.push_back(uNp1[0]), v.push_back(uNp1[1]), r.push_back(uNp1[2]);

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

  double getLateralDist() const {
	  // Compute perpendicular distance from path
	  const double thetaStart = getAngle(pathStart);
	  const double thetaEnd = getAngle(pathEnd);

	  const double distStart = getDistance(pathStart);
	  const double distEnd 	= getDistance(pathEnd); 

	  double retVal;
	  if( abs(thetaEnd) >= M_PI/2.0 && (thetaStart>=-M_PI/2.0 && thetaStart<=M_PI/2.0) )
	  {
		  retVal = distStart*abs(sin(thetaStart));
	  } else {
		  retVal = (distStart < distEnd) ? distStart : distEnd;
	  }

	  return retVal/params.l; // normalize with ship width
  }

  double getReward() const {
	  const double latDist = getLateralDist();
	  // angle wrt path - punish 45deg to be equal to 1 latDist, and linearly vary up and down
	  const double anglePenalty = abs(r.back())/(M_PI/4.0);
	  return -(latDist + anglePenalty);
  }

  bool checkTermination() {
	  const double threshold= 0.1*this->params.l; // within 10% of ship width
	  const double distGoal = getDistance(pathEnd);
	  const bool goal = (distGoal < threshold) ? true : false;
	  if (goal) isOver = true; 

	  // Check if need to abort
	  const double latDist = getLateralDist(); // already normalized by params.l
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
  const double xPathEnd[2] = {50,0};

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

	    /*// Overwrite actions
	      actions[0] = -20.0; // Thruster Left
	      actions[1] = 20.0; // Thruster Right*/

	    // Magnify order(1) actions to order(10) forces
	    thrustL.push_back(50*actions[0]);
	    thrustR.push_back(50*actions[1]);
	    forceX[step] = actions[0] + actions[1];

	    boat.advance();
	    tt.push_back(dt*(step+1));

	    const bool abortSim = boat.checkTermination();// Abort if too far from trajectory
	    if(boat.is_over()){ // Terminate simulation 

		    const double finalReward = 10*boat.params.l;

		    if(abortSim){
			    comm.sendTermState(boat.getState(), -finalReward, 0);
			    printf("Sim #%d reporting that boat got lost.\n", sim); fflush(NULL);
		    } else {
			    comm.sendTermState(boat.getState(), +finalReward, 0);
			    printf("Sim #%d reporting that boat reached safe harbor.\n", sim+1); fflush(NULL);
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
    fprintf(temp, "time \t x \t y \t theta \t u \t v \t r \t thrustL \t thrustR \t forceX \t forceY \t torque\n");
    for(unsigned int i=0; i<tt.size(); i++)  fprintf(temp, "%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n", tt[i], x[i], y[i], thetaR[i], u[i], v[i], r[i], thrustL[i], thrustR[i], forceX[i], forceY, torque[i]);
    fclose(temp);

  }
  return 0;
}
