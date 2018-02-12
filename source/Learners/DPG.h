/*
 *  DPG.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner_offPolicy.h"

class DPG : public Learner_offPolicy
{
  Aggregator* relay;
  const Uint nA = env->aI.dim;
  vector<Rvec> OrUhState = vector<Rvec>(nAgents,Rvec(nA,0));
  void Train_BPTT(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

public:
  DPG(Environment*const env, Settings & settings);
  void select(const Agent& agent) override;
};
