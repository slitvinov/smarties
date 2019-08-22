//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner.h"

namespace pybind11
{
  class object;
}

class Learner_pytorch: public Learner
{
 public:
  const bool bSampleSequences = settings.bSampleSequences;
  // hyper-parameters:
  const Uint batchSize = settings.batchSize_loc;
  const Uint ESpopSize = settings.ESpopSize;
  const Real learnR = settings.learnrate;
  const Real explNoise = settings.explNoise;

 protected:
 
  void spawnTrainTasks();
  // auto Nets;
  // pybind11::object Nets;
  // pybind11::object * Nets;
  // std::vector<pybind11::object*> Nets;
  std::vector<pybind11::object> Nets;

 public:
  Learner_pytorch(Environment*const env, Settings & settings);

  void select(Agent& ) override;
  void setupTasks(TaskQueue& tasks) override;
  virtual ~Learner_pytorch();

  virtual void getMetrics(std::ostringstream& buff) const override;
  virtual void getHeaders(std::ostringstream& buff) const override;
  virtual void save() override;
  virtual void restart() override;

};


