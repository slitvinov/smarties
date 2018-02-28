/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner_offPolicy.h"
#include "../Math/Quadratic_advantage.h"

class NAF : public Learner_offPolicy
{
  //Network produces a vector. The two following vectors specify:
  // - the sizes of the elements that compose the vector
  // - the starting indices along the output vector of each
  const vector<Uint> net_outputs = {1, compute_nL(aInfo.dim), aInfo.dim};
  const vector<Uint> net_indices = {0, 1, 1+compute_nL(aInfo.dim)};

  inline Quadratic_advantage prepare_advantage(const Rvec& out) const
  {
    return Quadratic_advantage(vector<Uint>{net_indices[1], net_indices[2]}, &aInfo, out);
  }

  void Train_BPTT(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

public:
  NAF(Environment*const env, Settings & settings);
  void select(const Agent& agent) override;
  void test();
  static inline Uint compute_nL(const Uint NA)
  {
    return (NA*NA + NA)/2;
  }
};
