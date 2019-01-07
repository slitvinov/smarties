//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Diego Rossinelli.
//

#pragma once

#include <string>
#include <unordered_map>

class Timer;

struct Timings
{
  bool started;
  int iterations;
  int64_t total;
  Timer * timer;

  Timings();
  ~Timings();
};

class Profiler
{
 public:
  enum Unit {s, ms, us};

 private:
  std::unordered_map<std::string, Timings>  timings;

  std::string ongoing;
  int numStarted;

  std::string __printStatAndReset(Unit unit, std::string prefix);

 public:
  Profiler();

  void start(std::string name);

  void stop();

  void stop_start(std::string name);

  double elapsed(std::string name, Unit unit = Unit::ms);

  std::string printStatAndReset(Unit unit = Unit::ms);

  void reset();
};
