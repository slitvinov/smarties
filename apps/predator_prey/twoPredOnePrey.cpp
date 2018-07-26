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
#define SAVEFREQ 2000
#define STEPFREQ 1
//#define PERIODIC
#define COOPERATIVE
//#define PLOT_TRAJ

#ifdef PLOT_TRAJ
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class Window
{
 private:
	static constexpr int plotDataSize = 500;
	std::vector<double> xData1 = std::vector<double>(plotDataSize, 0);
	std::vector<double> yData1 = std::vector<double>(plotDataSize, 0);
	std::vector<double> xData2 = std::vector<double>(plotDataSize, 0);
	std::vector<double> yData2 = std::vector<double>(plotDataSize, 0);
	std::vector<double> xData3 = std::vector<double>(plotDataSize, 0);
	std::vector<double> yData3 = std::vector<double>(plotDataSize, 0);

 public:
	Window() {
    plt::figure();
    plt::figure_size(320, 320);
  }

  void update(int step, int sim, std::array<double, 2> x1, std::array<double, 2> x2, std::array<double, 2> x3)
  {
    //printf("%d %g %g %g %g\n", step, x1, y1, x2, y2); fflush(0);
    if(sim % SAVEFREQ || step % STEPFREQ) return;
    if(step>plotDataSize) step = plotDataSize;
    std::fill(xData1.data() + step, xData1.data() + plotDataSize, x1[0]);
    std::fill(yData1.data() + step, yData1.data() + plotDataSize, x1[1]);
    std::fill(xData2.data() + step, xData2.data() + plotDataSize, x2[0]);
    std::fill(yData2.data() + step, yData2.data() + plotDataSize, x2[1]);
    std::fill(xData3.data() + step, xData3.data() + plotDataSize, x3[0]);
    std::fill(yData3.data() + step, yData3.data() + plotDataSize, x3[1]);
    plt::clf();
    plt::xlim(-0.1, 1.1);
    plt::ylim(-0.1, 1.1);
    plt::plot(xData1, yData1, "r-");
    plt::plot(xData2, yData2, "m-");
    plt::plot(xData3, yData3, "b-");
    std::vector<double> X1(1,x1[0]), Y1(1,x1[1]), X2(1,x2[0]), Y2(1,x2[1]), X3(1,x3[0]), Y3(1,x3[1]);
    plt::plot(X1, Y1, "or");
    plt::plot(X2, Y2, "om");
    plt::plot(X3, Y3, "ob");
    //plt::show(false);
    char temp[32]; 
    sprintf(temp, "%05d", step);
    std::string temp2 = temp;
    plt::save("./"+std::to_string(sim)+"_"+temp2+".png");
  }
};
#endif

using namespace std;

struct Entity
{
  //const unsigned nQuadrants; // NOTE: not used at the moment. Should we just stick to angles??
  const unsigned nStates;
  const double maxSpeed;
  std::mt19937& genA;
  bool isOver=false;

  Entity(std::mt19937& _gen, const unsigned nQ, const double vM)
    : nStates(nQ), maxSpeed(vM), genA(_gen) {}

  array<double, 2> p;
  double speed; 

  void reset() {
    uniform_real_distribution<double> distrib(0, EXTENT);
    p[0] = distrib(genA);
    p[1] = distrib(genA);
    speed = maxSpeed;
	isOver = false;
	}

  bool is_over() {
    return isOver; // TODO add catching condition - EHH??
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
    #ifdef PERIODIC
      if (p[0] >= EXTENT) p[0] -= EXTENT;
      if (p[0] <  0)      p[0] += EXTENT;
      if (p[1] >= EXTENT) p[1] -= EXTENT;
      if (p[1] <  0)      p[1] += EXTENT;
    #else
      if (p[0] > EXTENT) p[0] = EXTENT;
      if (p[0] < 0)      p[0] = 0;
      if (p[1] > EXTENT) p[1] = EXTENT;
      if (p[1] < 0)      p[1] = 0;
    #endif
  }

  template<typename T>
  unsigned getAngle(const T& E) const {
    double relX = E.p[0] - p[0];
    double relY = E.p[1] - p[1];
    #ifdef PERIODIC
      if(relX >  EXTENT/2) relX -= EXTENT; // WUT????
      if(relX < -EXTENT/2) relX += EXTENT;
      if(relY >  EXTENT/2) relY -= EXTENT;
      if(relY < -EXTENT/2) relY += EXTENT;
    #endif
    return std::atan2(relY, relX);
  }

  template<typename T>
  double getDistance(const T& E) const {
    double relX = E.p[0] - p[0];
    double relY = E.p[1] - p[1];
    #ifdef PERIODIC
      if(relX >  EXTENT/2) relX -= EXTENT;
      if(relX < -EXTENT/2) relX += EXTENT;
      if(relY >  EXTENT/2) relY -= EXTENT;
      if(relY < -EXTENT/2) relY += EXTENT;
    #endif
    return std::sqrt(relX*relX + relY*relY);
  }
};


struct Prey: public Entity
{
  const double stdNoise; // Only prey assumed to suffer from noise

  Prey(std::mt19937& _gen, const unsigned _nStates, const double vM, const double dN)
    : Entity(_gen, _nStates, vM), stdNoise(dN) {}

  template<typename T>
  vector<double> getState(const T& E1, const T& E2) { // wrt enemy E
    vector<double> state(nStates, 0);
    state[0] = p[0];
    state[1] = p[1];
    const double angEnemy1 = getAngle(E1);
    const double angEnemy2 = getAngle(E2);
    const double distEnemy1 = getDistance(E1);
    const double distEnemy2 = getDistance(E2);
    const double distEnemy = distEnemy1*distEnemy2;
    // close? low noise. moving slow? low noise
    //const double noiseAmp = stdNoise*distEnemy*speed/std::pow(maxSpeed,2);
    // or, can also think of it as ETA (Estimated Time of Arrival)
    const double ETA = distEnemy/fabs(speed);
    const double noiseAmp = stdNoise*ETA;
    std::normal_distribution<double> distrib(0, noiseAmp); // mean=0, stdDev=noiseAmp
    const double noisyAng1 = angEnemy1 + distrib(genA);
    const double noisyAng2 = angEnemy2 + distrib(genA);
    state[2] = distEnemy1*std::cos(noisyAng1); 
    state[3] = distEnemy1*std::sin(noisyAng1);
    state[4] = distEnemy2*std::cos(noisyAng2);
    state[5] = distEnemy2*std::sin(noisyAng2);
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

struct Predator: public Entity
{
  const double velPenalty;
  Predator(std::mt19937& _gen, const unsigned _nStates, const double vM, const double vP)
    : Entity(_gen, _nStates, vP*vM), velPenalty(vP) {}

  template<typename T1, typename T2>
  vector<double> getState(const T1& _prey, const T2& _pred) const { // wrt enemy (or adversary) E
    vector<double> state(nStates, 0);
    state[0] = p[0];
    state[1] = p[1];
    const double angPrey = getAngle(_prey); // No noisy angle for predator
    const double angPred = getAngle(_pred); // No noisy angle for predator
    const double distPrey = getDistance(_prey);
    const double distPred = getDistance(_pred);
    state[2] = distPrey*std::cos(angPrey);
    state[3] = distPrey*std::sin(angPrey);
    state[4] = distPred*std::cos(angPred);
    state[5] = distPred*std::sin(angPred);
    return state;
  }

  template<typename T1, typename T2>
  double getReward(const T1& _prey, const T2& _pred) const {
#ifdef COOPERATIVE
    const double distMult = _prey.getDistance(*this) * _prey.getDistance(_pred);
    return - distMult; // cooperative predators
#else
    return - getDistance(_prey); // competitive predators
#endif
  }
};

int main(int argc, const char * argv[])
{
  //communication:
  const int socket = std::stoi(argv[1]);
  const unsigned maxStep = 500;
  const int control_vars = 2; // 2 components of velocity
  const int state_vars = 6;   // number of states (self pos[2], enemy cos|sin[theta])
  const int number_of_agents = 3; // 2 predator, 1 prey
  //Sim box has size EXTENT. Fraction of box that agent can traverse in 1 step:
  const double maxSpeed = 0.02 * EXTENT/dt;
  //socket number is given by RL as first argument of execution
  Communicator comm(socket, state_vars, control_vars, number_of_agents);

  // predator last arg is how much slower than prey (eg 50%)
  Predator pred1(comm.gen, state_vars, maxSpeed, 0.5);
  Predator pred2(comm.gen, state_vars, maxSpeed, 0.5);
  // prey last arg is observation noise (eg ping of predator is in 1 stdev of noise)
  // Prey     prey(comm.gen, state_vars, maxSpeed, 1.0); // The noise was large, the prey didn't run away quickly if preds were far away
  Prey     prey(comm.gen, state_vars, maxSpeed, 0.0);

#ifdef COOPERATIVE
  printf("Cooperative predators\n");
#else
  printf("Competitive predators\n");
#endif
  fflush(NULL);

#ifdef PLOT_TRAJ
  Window plot;
#endif

  unsigned sim = 0;
  while(true) //train loop
  {
    //reset environment:
    pred1.reset();
    pred2.reset();
    prey.reset();

    //send initial state
    comm.sendInitState(pred1.getState(prey,pred2), 0);
    comm.sendInitState(pred2.getState(prey,pred1), 1);
    comm.sendInitState(prey.getState(pred1,pred2), 2);

    unsigned step = 0;
    while (true) //simulation loop
    {
      pred1.advance(comm.recvAction(0));
      pred2.advance(comm.recvAction(1));
      prey.advance(comm.recvAction(2));

	  vector<bool> gotCaught = prey.checkTermination(pred1,pred2);
	  if(prey.is_over()){ // Terminate simulation 
		  // Cooperative hunting - both predators get reward
		  const double finalReward = 10*EXTENT;

#ifdef COOPERATIVE
		  comm.sendTermState(pred1.getState(prey,pred2), finalReward, 0);
		  comm.sendTermState(pred2.getState(prey,pred1), finalReward, 1);
#else
		  // Competitive hunting - only one winner, other one gets jack (also, change the reward to be not d1*d2, but just d_i if use competitive)
          comm.sendTermState(pred1.getState(prey,pred2), finalReward*gotCaught[0], 0);
          comm.sendTermState(pred2.getState(prey,pred1), finalReward*gotCaught[1], 1);
#endif
		  comm.sendTermState(prey.getState(pred1,pred2),-finalReward, 2);

		  printf("Sim #%d reporting that prey got its world rocked.\n", sim); fflush(NULL);
		  sim++; break;
	  }

#ifdef PLOT_TRAJ
      plot.update(step, sim, pred1.p, pred2.p, prey.p);
#endif

      if(step++ < maxStep)
      {
        comm.sendState(  pred1.getState(prey,pred2), pred1.getReward(prey,pred2), 0);
        comm.sendState(  pred2.getState(prey,pred1), pred2.getReward(prey,pred1), 1);
        comm.sendState(  prey.getState(pred1,pred2), prey.getReward(pred1,pred2), 2);
      }
      else
      {
        comm.truncateSeq(pred1.getState(prey,pred2), pred1.getReward(prey,pred2), 0);
        comm.truncateSeq(pred2.getState(prey,pred1), pred2.getReward(prey,pred2), 1);
        comm.truncateSeq(prey.getState(pred1,pred2), prey.getReward(pred1,pred2), 2);
        sim++;
        break;
      }
    }
  }
  return 0;
}
