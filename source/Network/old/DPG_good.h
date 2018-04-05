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
  const Real tgtFrac = 0.1, learnR;
  //const Real OrUhDecay = .85; // as in original
  const Real OrUhDecay = 0; // no correlated noise
  vector<Rvec> OrUhState = vector<Rvec>(nAgents,Rvec(nA,0));

  Real beta = 0.2;
  MPI_Request nData_request = MPI_REQUEST_NULL;
  double ndata_reduce_result[2], ndata_partial_sum[2];

  void Train_BPTT(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

public:
  DPG(Environment*const env, Settings & settings);
  void select(const Agent& agent) override;
  void prepareGradient() override;
  void getMetrics(ostringstream& buff) const;
  void getHeaders(ostringstream& buff) const;
};
