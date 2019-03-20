//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Learner.h"

class Approximator;
class Encapsulator;

class Learner_approximator: public Learner
{
 public:
  const bool bSampleSequences = settings.bSampleSequences;
  // hyper-parameters:
  const Uint batchSize = settings.batchSize_loc;
  const Uint ESpopSize = settings.ESpopSize;
  const Real learnR = settings.learnrate;
  const Real explNoise = settings.explNoise;

 protected:
  Encapsulator * const input;
  std::vector<Approximator*> F;

  void createSharedEncoder(const Uint privateNum = 1);
  bool predefinedNetwork(Builder& input_net, const Uint privateNum = 1);

  virtual void TrainBySequences(const Uint seq, const Uint wID,
    const Uint bID, const Uint tID) const = 0;
  virtual void Train(const Uint seq, const Uint samp, const Uint wID,
    const Uint bID, const Uint tID) const = 0;
  void spawnTrainTasks();
  virtual void prepareGradient();
  virtual void applyGradient();

 public:
  Learner_approximator(Environment*const env, Settings & settings);

  virtual ~Learner_approximator();

  virtual void getMetrics(std::ostringstream& buff) const override;
  virtual void getHeaders(std::ostringstream& buff) const override;
  virtual void save() override;
  virtual void restart() override;
};
