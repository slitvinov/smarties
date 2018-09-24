//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#pragma once
#include "Parameters.h"
#include <iomanip>

class Optimizer
{
 protected:
  const MPI_Comm mastersComm;
  const Uint learn_size, pop_size;
  const Parameters * const weights;
  const Parameters * const tgt_weights;
  const Uint pDim = weights->nParams;
  const Real eta_init;
  Uint cntUpdateDelay = 0, totGrads = 0;

 public:
  bool bAnnealLearnRate = true;
  const Real lambda, epsAnneal, tgtUpdateAlpha;
  long unsigned nStep = 0;
  nnReal eta = eta_init;

  Optimizer(Settings&S, const Parameters*const W, const Parameters*const W_TGT):
  mastersComm(S.mastersComm), learn_size(S.learner_size), pop_size(S.ESpopSize),
  weights(W), tgt_weights(W_TGT), eta_init(S.learnrate), lambda(S.nnLambda),
  epsAnneal(S.epsAnneal), tgtUpdateAlpha(S.targetDelay) {}

  virtual ~Optimizer() {}
  virtual void save(const string fname, const bool bBackup) = 0;
  virtual int restart(const string fname) = 0;

  virtual void prepare_update(const int BS, const Rvec&L) = 0;
  virtual void apply_update() = 0;

  virtual void getMetrics(ostringstream& buff) = 0;
  virtual void getHeaders(ostringstream& buff) = 0;
};

class AdamOptimizer : public Optimizer
{
 protected:
  const Real beta_1, beta_2;
  Real beta_t_1 = beta_1, beta_t_2 = beta_2;
  const Parameters * const gradSum = weights->allocateGrad();
  const Parameters * const _1stMom = weights->allocateGrad();
  const Parameters * const _2ndMom = weights->allocateGrad();
  const Parameters * const _2ndMax = weights->allocateGrad();
  vector<std::mt19937>& generators;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  MPI_Request batchRequest = MPI_REQUEST_NULL;
  //const Real alpha_eSGD, gamma_eSGD, eta_eSGD, eps_eSGD, delay;
  //const Uint L_eSGD;
  //nnReal *const _muW_eSGD, *const _muB_eSGD;
  const vector<Parameters*> grads;

 public:

  AdamOptimizer(Settings&S, const Parameters*const W, const Parameters*const WT,
    const vector<Parameters*>&G, const Real B1=.9, const Real B2=.999) :
    Optimizer(S, W, WT), beta_1(B1), beta_2(B2), generators(S.generators),
    grads(G) { }
  //alpha_eSGD(0.75), gamma_eSGD(10.), eta_eSGD(.1/_s.targetDelay),
  //eps_eSGD(1e-3), delay(_s.targetDelay), L_eSGD(_s.targetDelay),
  //_muW_eSGD(initClean(nWeights)), _muB_eSGD(initClean(nBiases))

  ~AdamOptimizer() {
   _dispose_object(gradSum); _dispose_object(_1stMom);
   _dispose_object(_2ndMom); _dispose_object(_2ndMax);
  }

  void prepare_update(const int BS, const Rvec& L) override;
  void apply_update() override;

  void save(const string fname, const bool bBackup) override;
  int restart(const string fname) override;
  void getMetrics(ostringstream& buff) override {}
  void getHeaders(ostringstream& buff) override {}
};
