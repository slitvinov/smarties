/*
 *  DQN.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner_offPolicy.h"


class DQN : public Learner_offPolicy
{
  void TrainBySequences(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

public:
  DQN(Environment*const env, Settings & settings);
  void select(const Agent& agent) override;
};
