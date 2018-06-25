
#include <iostream>
#include <cmath>
#include <random>
#include <cstdio>
#include <vector>
#include <array>
#include <functional>
#include "Communicator.h"

#define EXTENT 1
#define SAVEFREQ 1000
#define STEPFREQ 10

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
    plt::xlim(0, 1);
    plt::ylim(0, 1);
  }

  void update(int step, int sim, double x1, double y1, double x2, double y2)
  {
    //printf("%d %g %g %g %g\n", step, x1, y1, x2, y2); fflush(0);
    if(sim % SAVEFREQ || step % STEPFREQ) return;
    if(step>plotDataSize) step = plotDataSize;
    std::fill(xData1.data() + step, xData1.data() + plotDataSize, x1);
    std::fill(yData1.data() + step, yData1.data() + plotDataSize, y1);
    std::fill(xData2.data() + step, xData2.data() + plotDataSize, x2);
    std::fill(yData2.data() + step, yData2.data() + plotDataSize, y2);
    plt::plot(xData1, yData1, "b-");
    plt::plot(xData2, yData2, "r-");
    //plt::show(false);
    plt::save("./"+std::to_string(sim)+"_"+std::to_string(step)+".png");
  }
};

using namespace std;

struct Entity
{
  const unsigned nQuadrants;
  const double velMagnitude;
  Entity(const unsigned nQ, const double vM)
    : nQuadrants(nQ), velMagnitude(vM) {}

  array<double, 2> p;
  double actScal;

  void reset(std::mt19937& gen) {
    uniform_real_distribution<double> dist(0, EXTENT);
    p[0] = dist(gen);
    p[1] = dist(gen);
    actScal = 1; // so that prey overwrites background
	}

  bool is_over() {
    return false; // TODO add catching condition
  }

  int advance(vector<double> act) {
    assert(act.size() == 2);
    actScal = std::sqrt(act[0]*act[0] + act[1]*act[1]) / velMagnitude;
    if( actScal > 1)
    {
      p[0] += act[0] * velMagnitude / actScal;
      p[1] += act[1] * velMagnitude / actScal;
      actScal = 1;
    }
    else
    {
      p[0] += act[0];
      p[1] += act[1];
    }

    if (p[0] >= EXTENT) p[0] -= EXTENT;
    if (p[0] <  0)      p[0] += EXTENT;
    if (p[1] >= EXTENT) p[1] -= EXTENT;
    if (p[1] <  0)      p[1] += EXTENT;
    return is_over();
  }

  template<typename T>
  unsigned getQuadrant(const T& E) const {
    const double relX = E.p[0] - p[0];
    const double relY = E.p[1] - p[1];
    const double relA = std::atan2(relY, relX) + M_PI; // between 0 and 2pi
    assert(relA >= 0 and relA <= 2*M_PI);
    return nQuadrants*relA/(2*M_PI + 2.2e-16);
  }
};

struct Prey: public Entity
{
  const double stdNoise;
  vector<double> background = vector<double>(nQuadrants, 0);

  void updateBackground(std::mt19937& gen, const double fac)
  {
    normal_distribution<double> dist(0, stdNoise);
    for (unsigned i=0; i<nQuadrants; i++)
      background[i] = fac*dist(gen) + (1-fac)*background[i];
  }

  Prey(const unsigned nQ, const double vM, const double dN)
    : Entity(nQ, vM), stdNoise(dN) {}

  template<typename T>
  vector<double> getState(const T& E, std::mt19937& gen) {
    updateBackground(gen, actScal);
    vector<double> state = background;
    const unsigned quadEnemy = getQuadrant(E);
    state[quadEnemy] = std::max(1., state[quadEnemy]);
    return state;
  }

  template<typename T>
  double getReward(const T& E) const {
    const double relX = E.p[0] - p[0];
    const double relY = E.p[1] - p[1];
    return std::sqrt(relX*relX + relY*relY);
  }
};

struct Predator: public Entity
{
  const double velPenalty;
  Predator(const unsigned nQ, const double vM, const double vP)
    : Entity(nQ, vP*vM), velPenalty(vP) {}

  template<typename T>
  vector<double> getState(const T& E) const {
    vector<double> state(nQuadrants, 0);
    const unsigned quadEnemy = getQuadrant(E);
    state[quadEnemy] = 1;
    return state;
  }

  template<typename T>
  double getReward(const T& E) const {
    const double relX = E.p[0] - p[0];
    const double relY = E.p[1] - p[1];
    return -std::sqrt(relX*relX + relY*relY);
  }
};

int main(int argc, const char * argv[])
{
  //communication:
  const int socket = std::stoi(argv[1]);
  const unsigned maxStep = 500;
  const int control_vars = 2; // 2 components of velocity
  const int state_vars = 8;   // number of sensor quadrants
  const int number_of_agents = 2; // predator prey
  //Sim box has size EXTENT. Fraction of box that agent can traverse in 1 step:
  const double velScale = 0.02 * EXTENT;
  //socket number is given by RL as first argument of execution
  Communicator comm(socket, state_vars, control_vars, number_of_agents);

  // predator additional arg is how much slower than prey (eg 50%)
  Predator pred(state_vars, velScale, 0.5);
  // prey arg is observation noise (eg ping of predator is in 1 stdev of noise)
  Prey     prey(state_vars, velScale, 1.0);

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
