#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <array>
#include <functional>
#include "Communicator.h"

#define EXTENT 1.0
#define dt 1.0
#define SAVEFREQ 1
#define STEPFREQ 1
//#define PERIODIC
#define COOPERATIVE
#define PLOT_TRAJ

using namespace std;

struct Entity
{
  const unsigned nStates;
  const double length;
  //const double maxSpeed, length;
  const double startX[2], endX[2];
  std::mt19937& genA;
  bool isOver=false;

  Entity(std::mt19937& _gen, const unsigned nQ, const double vM)
    : nStates(nQ), maxSpeed(vM), genA(_gen) {}

  array<double, 2> p;
  double speed; 

  void reset() {
    normal_distribution<double> distrib(0, 2*length);
    p[0] = this->startX[0] + distrib(genA);
    p[1] = this->startX[1] + distrib(genA);
    speed = maxSpeed;
	isOver = false;
	}

  bool is_over() {
    return isOver; 
  }

  void advance(vector<double> act) {
    assert(act.size() == 2);
    speed = std::sqrt(act[0]*act[0] + act[1]*act[1]);

    if( speed > maxSpeed) { // Rescale the u and v components so that speed = maxSpeed
      p[0] += act[0] * (maxSpeed / speed) *dt;
      p[1] += act[1] * (maxSpeed / speed) *dt;
      speed = maxSpeed;
    } else {
      p[0] += act[0] *dt;
      p[1] += act[1] *dt;
    }
      if (p[0] > EXTENT) p[0] = EXTENT;
      if (p[0] < 0)      p[0] = 0;
      if (p[1] > EXTENT) p[1] = EXTENT;
      if (p[1] < 0)      p[1] = 0;
  }

  template<typename T>
  unsigned getAngle(const T& E) const {
    double relX = E.p[0] - p[0];
    double relY = E.p[1] - p[1];
    return std::atan2(relY, relX);
  }

  template<typename T>
  double getDistance(const T& E) const {
    double relX = E.p[0] - p[0];
    double relY = E.p[1] - p[1];
    return std::sqrt(relX*relX + relY*relY);
  }
};


struct SUV: public Entity
{
  const double stdNoise; // Only nemo assumed to suffer from noise

  SUV(std::mt19937& _gen, const unsigned _nStates, const double vM, const double dN)
    : Entity(_gen, _nStates, vM), stdNoise(dN) {}

  template<typename T>
  vector<double> getState(const T& E1, const T& E2) { // wrt enemy E
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
    state[3] = distEnemy1*std::sin(noisyAng1);
    return state;
  }

  template<typename T>
  double getReward(const T& E1, const T& E2) const {
    return getDistance(E1)*getDistance(E2);
  }

  template<typename T>
  vector<bool> checkTermination(const T& E1, const T& E2) {
	const double threshold= 0.01*EXTENT;
    const double dist1 = getDistance(E1);
    const double dist2 = getDistance(E2);
	const bool caught1 = (dist1 < threshold) ? true : false;
	const bool caught2 = (dist2 < threshold) ? true : false;
	const vector<bool> gotCaught = {caught1, caught2};

	if(caught1 || caught2) isOver = true; 
	return gotCaught;
  }
 
};


int main(int argc, const char * argv[])
{
  //communication:
  const int socket = std::stoi(argv[1]);
  const unsigned maxStep = 500;
  const int control_vars = 2; // 2 components of velocity
  const int state_vars = 6;   // number of states (self pos[2], enemy cos|sin[theta])
  const int number_of_agents = 3; // 2 predator, 1 nemo
  //Sim box has size EXTENT. Fraction of box that agent can traverse in 1 step:
  const double maxSpeed = 0.02 * EXTENT/dt;
  //socket number is given by RL as first argument of execution
  Communicator comm(socket, state_vars, control_vars, number_of_agents);

  SUV     nemo(comm.gen, state_vars, maxSpeed, 0.0);

  unsigned sim = 0;
  while(true) //train loop
  {
    //reset environment:
    nemo.reset();

    //send initial state
    comm.sendInitState(pred1.getState(nemo,pred2), 0);

    unsigned step = 0;
    while (true) //simulation loop
    {
      nemo.advance(comm.recvAction(0));

	  vector<bool> gotCaught = nemo.checkTermination(pred1,pred2);
	  if(nemo.is_over()){ // Terminate simulation 
		  const double finalReward = 10*EXTENT;
		  comm.sendTermState(nemo.getState(pred1,pred2),-finalReward, 2);

		  printf("Sim #%d reporting that nemo got its world rocked.\n", sim); fflush(NULL);
		  sim++; break;
	  }

      if(step++ < maxStep)
      {
        comm.sendState(  nemo.getState(pred1,pred2), nemo.getReward(pred1,pred2), 2);
      }
      else
      {
        comm.truncateSeq(nemo.getState(pred1,pred2), nemo.getReward(pred1,pred2), 2);
        sim++;
        break;
      }
    }
  }
  return 0;
}
