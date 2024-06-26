#include "smarties.h"
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

template <typename Func, typename Vec>
Vec rk46_nl(double t0, double dt, Vec u0, Func &&Diff) {
  static constexpr double a[] = {0.000000000000,  -0.737101392796,
                                 -1.634740794341, -0.744739003780,
                                 -1.469897351522, -2.813971388035};
  static constexpr double b[] = {0.032918605146, 0.823256998200,
                                 0.381530948900, 0.200092213184,
                                 1.718581042715, 0.270000000000};
  static constexpr double c[] = {0.000000000000, 0.032918605146,
                                 0.249351723343, 0.466911705055,
                                 0.582030414044, 0.847252983783};
  static constexpr int s = 6;
  Vec w;
  Vec u(u0);
  double t;

  for (int i = 0; i < s; ++i) {
    t = t0 + dt * c[i];
    w = w * a[i] + Diff(u, t) * dt;
    u = u + w * b[i];
  }
  return u;
}

struct Vec4 {
  double y1, y2, y3, y4;

  Vec4(double _y1 = 0, double _y2 = 0, double _y3 = 0, double _y4 = 0)
      : y1(_y1), y2(_y2), y3(_y3), y4(_y4){};

  Vec4 operator*(double v) const {
    return Vec4(y1 * v, y2 * v, y3 * v, y4 * v);
  }

  Vec4 operator+(const Vec4 &v) const {
    return Vec4(y1 + v.y1, y2 + v.y2, y3 + v.y3, y4 + v.y4);
  }
};

struct CartPole {
  const double mp = 0.1;
  const double mc = 1;
  const double l = 0.5;
  const double g = 9.81;
  const double dt = 4e-4;
  const int nsteps = 50;
  int step = 0;
  Vec4 u;
  double F = 0, t = 0;
  void reset(std::mt19937 &gen) {
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    u = Vec4(dist(gen), dist(gen), dist(gen), dist(gen));
    step = 0;
    F = 0;
    t = 0;
  }

  bool is_failed() {
    return std::fabs(u.y1) > 2.4 || std::fabs(u.y3) > M_PI / 15;
  }
  bool is_over() {
    return step >= 500 || std::fabs(u.y1) > 2.4 || std::fabs(u.y3) > M_PI / 15;
  }
  int advance(std::vector<double> action) {
    F = action[0];
    step++;
    for (int i = 0; i < nsteps; i++) {
      u = rk46_nl(t, dt, u,
                  std::bind(&CartPole::Diff, this, std::placeholders::_1,
                            std::placeholders::_2));
      t += dt;
      if (is_over())
        return 1;
    }
    return 0;
  }

  std::vector<double> getState(const int size = 6) {
    assert(size == 4 || size == 6);
    std::vector<double> state(size);
    state[0] = u.y1;
    state[1] = u.y2;
    state[2] = u.y4;
    state[3] = u.y3;
    if (size == 6) {
      state[4] = std::cos(u.y3);
      state[5] = std::sin(u.y3);
    }
    return state;
  }

  double getReward() {
    return 1 - (std::fabs(u.y3) > M_PI / 15 || std::fabs(u.y1) > 2.4);
  }

  Vec4 Diff(Vec4 _u, double _t) {
    Vec4 res;

    const double cosy = std::cos(_u.y3), siny = std::sin(_u.y3);
    const double w = _u.y4;
    const double totMass = mp + mc;
    const double fac2 = l * (4.0 / 3 - (mp * cosy * cosy) / totMass);
    const double F1 = F + mp * l * w * w * siny;
    res.y4 = (g * siny - F1 * cosy / totMass) / fac2;
    res.y2 = (F1 - mp * l * res.y4 * cosy) / totMass;
    res.y1 = _u.y2;
    res.y3 = _u.y4;
    return res;
  }
};

inline void app_main(smarties::Communicator *const comm, int argc,
                     char **argv) {
  const int control_vars = 1; // force along x
  const int state_vars = 6;   // x, vel, angvel, angle, cosine, sine
  comm->setStateActionDims(state_vars, control_vars);

  // OPTIONAL: action bounds
  bool bounded = true;
  std::vector<double> upper_action_bound{10}, lower_action_bound{-10};
  comm->setActionScales(upper_action_bound, lower_action_bound, bounded);

  /*
    ALTERNATIVE for discrete actions:
    vector<int> n_options = vector<int>{2};
    comm->set_action_options(n_options);

    OPTIONAL: hide state variables. e.g. show cosine/sine but not angle
  */
  std::vector<bool> b_observable = {true, true, true, false, true, true};
  // std::vector<bool> b_observable = {true, false, false, false, true, true};
  comm->setStateObservable(b_observable);
  // comm->setIsPartiallyObservable();
  CartPole env;
  while (true) {
    env.reset(comm->getPRNG());
    if (comm->terminateTraining()) {
      return;
    }
    comm->sendInitState(env.getState());
    while (true) {
      if (comm->terminateTraining()) {
        return;
      }
      std::vector<double> action = comm->recvAction();
      bool poleFallen = env.advance(action);
      std::vector<double> state = env.getState();
      double reward = env.getReward();
      if (comm->terminateTraining()) {
        return;
      }
      if (poleFallen) {
        comm->sendTermState(state, reward);
        break;
      } else
        comm->sendState(state, reward);
    }
  }
}

int main(int argc, char **argv) {
  smarties::Engine e(argc, argv);
  if (e.parse())
    return 1;
  e.run(app_main);
  return 0;
}

//  Created by Dmitry Alexeev on 04/06/15.
//  Copyright (c) 2015 Dmitry Alexeev. All rights reserved.
// Julien Berland, Christophe Bogey, Christophe Bailly,
// Low-dissipation and low-dispersion fourth-order Runge-Kutta algorithm,
// Computers & Fluids, Volume 35, Issue 10, December 2006, Pages 1459-1463, ISSN
// 0045-7930, http://dx.doi.org/10.1016/j.compfluid.2005.04.003
