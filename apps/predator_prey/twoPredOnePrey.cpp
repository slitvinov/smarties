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
#define SAVEFREQ 1000
#define STEPFREQ 1
//#define PERIODIC

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

 public:
	Window() {
    plt::figure();
    plt::figure_size(320, 320);
  }

  void update(int step, int sim, double x1, double y1,
              double x2, double y2)
  {
    //printf("%d %g %g %g %g\n", step, x1, y1, x2, y2); fflush(0);
    if(sim % SAVEFREQ || step % STEPFREQ) return;
    if(step>plotDataSize) step = plotDataSize;
    std::fill(xData1.data() + step, xData1.data() + plotDataSize, x1);
    std::fill(yData1.data() + step, yData1.data() + plotDataSize, y1);
    std::fill(xData2.data() + step, xData2.data() + plotDataSize, x2);
    std::fill(yData2.data() + step, yData2.data() + plotDataSize, y2);
    plt::clf();
    plt::xlim(-0.1, 1.1);
    plt::ylim(-0.1, 1.1);
    plt::plot(xData1, yData1, "r-");
    plt::plot(xData2, yData2, "b-");
    std::vector<double> X1(1,x1), Y1(1,y1), X2(1,x2), Y2(1,y2);
    plt::plot(X1, Y1, "or");
    plt::plot(X2, Y2, "ob");
    //plt::show(false);
    plt::save("./"+std::to_string(sim)+"_"+std::to_string(step)+".png");
  }
};

using namespace std;

struct Entity
{
  const unsigned nQuadrants;
  const double maxSpeed;
  Entity(const unsigned nQ, const double vM)
    : nQuadrants(nQ), maxSpeed(vM) {}

  array<double, 2> p;
  double speed; // Scaling factor, to make the predator go slower than prey

  void reset(std::mt19937& gen) {
    uniform_real_distribution<double> distrib(0, EXTENT);
    p[0] = distrib(gen);
    p[1] = distrib(gen);
    speed = maxSpeed;
	}

  bool is_over() {
    return false; // TODO add catching condition - EHH??
  }

  int advance(vector<double> act) {
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
    return is_over();
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

  Prey(const unsigned nQ, const double vM, const double dN)
    : Entity(nQ, vM), stdNoise(dN) {}

  template<typename T>
  vector<double> getState(const T& E, std::mt19937& gen) { // wrt enemy E
    vector<double> state(4, 0);
    state[0] = p[0];
    state[1] = p[1];
    const double angEnemy = getAngle(E);
    const double distEnemy = getDistance(E);
    // close? low noise. moving slow? low noise
    //const double noiseAmp = stdNoise*distEnemy*speed/std::pow(maxSpeed,2);
    // or, can also think of it as ETA (Estimated Time of Arrival)
    const double ETA = distEnemy/fabs(speed);
    const double noiseAmp = stdNoise*ETA;
    std::normal_distribution<double> distrib(0, noiseAmp); // mean=0, stdDev=noiseAmp
    const double noisyAng = angEnemy + distrib(gen);
    state[2] = std::cos(noisyAng);
    state[3] = std::sin(noisyAng);
    return state;
  }

  template<typename T>
  double getReward(const T& E) const {
    return getDistance(E);
  }
};

struct Predator: public Entity
{
  const double velPenalty;
  Predator(const unsigned nQ, const double vM, const double vP)
    : Entity(nQ, vP*vM), velPenalty(vP) {}

  template<typename T>
  vector<double> getState(const T& E) const { // wrt enemy (or adversary) E
    vector<double> state(4, 0);
    state[0] = p[0];
    state[1] = p[1];
    const double angEnemy = getAngle(E); // No noisy angle for predator
    state[2] = std::cos(angEnemy);
    state[3] = std::sin(angEnemy);
    return state;
  }

  template<typename T>
  double getReward(const T& E) const {
    return - getDistance(E);
  }
};

int main(int argc, const char * argv[])
{
  //communication:
  const int socket = std::stoi(argv[1]);
  const unsigned maxStep = 500;
  const int control_vars = 2; // 2 components of velocity
  const int state_vars = 4;   // number of sensor quadrants
  const int number_of_agents = 2; // predator prey
  //Sim box has size EXTENT. Fraction of box that agent can traverse in 1 step:
  const double maxSpeed = 0.02 * EXTENT; // Again, assuming dt = 1
  //socket number is given by RL as first argument of execution
  Communicator comm(socket, state_vars, control_vars, number_of_agents);

  // predator last arg is how much slower than prey (eg 50%)
  Predator pred(state_vars, maxSpeed, 0.5);
  // prey last arg is observation noise (eg ping of predator is in 1 stdev of noise)
  Prey     prey(state_vars, maxSpeed, 1.0);

  Window plot;

  unsigned sim = 0;
  while(true) //train loop
  {
    //reset environment:
    pred.reset(comm.gen); //comm contains rng with different seed on each rank
    prey.reset(comm.gen); //comm contains rng with different seed on each rank

    //send initial state
    comm.sendInitState(pred.getState(prey),           0);
    comm.sendInitState(prey.getState(pred, comm.gen), 1);

    unsigned step = 0;
    while (true) //simulation loop
    {
      pred.advance(comm.recvAction(0));
      prey.advance(comm.recvAction(1));

      plot.update(step, sim, pred.p[0], pred.p[1], prey.p[0], prey.p[1]);

      if(step++ < maxStep)
      {
        comm.sendState(  pred.getState(prey),          pred.getReward(prey), 0);
        comm.sendState(  prey.getState(pred,comm.gen), prey.getReward(pred), 1);
      }
      else
      {
        comm.truncateSeq(pred.getState(prey),          pred.getReward(prey), 0);
        comm.truncateSeq(prey.getState(pred,comm.gen), prey.getReward(pred), 1);
        sim++;
        break;
      }
    }
  }
  return 0;
}
