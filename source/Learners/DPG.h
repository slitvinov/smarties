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

//#define LearnStDev

class DPG : public Learner_offPolicy
{
  Aggregator* relay;
  const Uint nA = env->aI.dim;
  const Real tgtFrac;
  const Real OrUhDecay = CmaxPol<=0? .85 : 0; // as in original
  //const Real OrUhDecay = 0; // no correlated noise
  vector<Rvec> OrUhState = vector<Rvec>(nAgents,Rvec(nA,0));

  ApproximateReductor<double, MPI_DOUBLE> reductor =
  ApproximateReductor<double, MPI_DOUBLE>(mastersComm, 2);

  inline Gaussian_policy prepare_policy(const Rvec& out,
    const Tuple*const t = nullptr) const {
    Gaussian_policy pol({0, nA}, &aInfo, out);
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }

  static inline Rvec clampGrad(Rvec grad_val, const Real val)
  {
    //if(grad_val[0]>0 && val>0) grad_val[0] /= 2;
    //if(grad_val[0]<0 && val<0) grad_val[0] /= 2;
    return grad_val;
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

public:
  DPG(Environment*const env, Settings & settings);
  void select(const Agent& agent) override;
  void prepareGradient() override;
  void getMetrics(ostringstream& buff) const;
  void getHeaders(ostringstream& buff) const;
};
