/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "Learner_onPolicy.h"
#include "../Math/Lognormal_policy.h"
#include "../Math/Gaussian_policy.h"

class GAE : public Learner_onPolicy
{
  const Real lambda = 0.95, DKL_target = 0.01, clip_fac = 0.2;
  //#ifdef INTEGRATEANDFIRESHARED
  //  const vector<Uint> net_outputs = {nA, 1};
  //#else
  //if order of this elements is changed class breaks!!
  //this becoz i use same function also to prepare stored old policy
  const vector<Uint> net_outputs = {nA,  1,   nA,      1};
  //#endif
  const vector<Uint> net_indices = { 0, nA, 1+nA, 1+2*nA};
  const Uint PenalID = net_indices[3], ValID = net_indices[1];

  //#ifdef INTEGRATEANDFIREMODEL
  //  inline Lognormal_policy prepare_policy(const vector<Real>& out) const
  //  {
  //    return Lognormal_policy(net_indices[0], net_indices[1], nA, out);
  //  }
  //#else
    inline Gaussian_policy prepare_policy(const vector<Real>& out) const
    {
      return Gaussian_policy(net_indices[0], net_indices[2], nA, out);
    }
    inline Gaussian_policy prepare_behavior(const vector<Real>& out) const
    {
      //stored policy is vector: {{action}, {stdev}}, so inds are hardcoded:
      return Gaussian_policy(0, nA, nA, out);
    }
  //#endif

  void Train_BPTT(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

public:
  GAE(MPI_Comm comm, Environment*const env, Settings & settings);

  //called by scheduler:
  void select(const int agentId, const Agent& agent) override;

  static Uint getnOutputs(const Uint NA)
  {
    //#ifdef INTEGRATEANDFIRESHARED
    //    return 2+NA;
    //#else
    return 2+NA+NA;
    //#endif
  }
};
