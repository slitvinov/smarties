/*
 *  Layers.h
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Parameters.h"
#include "saruprng.h"
#include <iomanip>


struct EntropyAdam {
  const nnReal eta, B1, B2;
  Saru& gen;
  EntropyAdam(const nnReal _eta, const nnReal beta1, const nnReal beta2,
    const nnReal betat1, const nnReal betat2, Saru& _gen) :
  eta(_eta*std::sqrt(1-betat2)/(1-betat1)), B1(beta1), B2(beta2), gen(_gen) {}

  inline nnReal step(const nnReal&grad, nnReal&M1, nnReal&M2, const nnReal fac){
    const nnReal DW  = grad * fac;
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = B2 * M2 + (1-B2) * DW*DW;
    M2 = M2<M1*M1? M1*M1 : M2; // gradient clipping to 1
    const nnReal _M2 = std::sqrt(M2 + nnEPS);
    //const nnReal ret = eta*M1/_M2;
    const nnReal ret = eta*(B1*M1 + (1-B1)*DW)/_M2;
    return ret + 1e-5*gen.d_mean0_var1(); //Adam plus exploratory noise
    //dest[i] += delay*(target[i]-dest[i]);
    //dest[i] += DW_ + RNG + eta_*gamma_eSGD*(target[i]-dest[i]);
    //_mu[i]  += alpha_eSGD*(dest[i] - _mu[i]);
  }
};

struct AdaMax {
  const nnReal eta, B1, B2;
  AdaMax(const nnReal _eta, const nnReal beta1, const nnReal beta2,
    const nnReal betat1, const nnReal betat2, Saru& _gen) :
  eta(_eta), B1(beta1), B2(beta2) {}

  inline nnReal step(const nnReal&grad, nnReal&M1, nnReal&M2, const nnReal fac){
    const nnReal DW  = grad * fac;
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = std::max(B2 * M2, std::fabs(DW));
    M2 = std::max(M2, nnEPS);
    //return eta*M1/M2;
    return eta*(B1*M1 + (1-B1)*DW)/M2;
  }
};

struct Adam {
  const nnReal eta, B1, B2;
  Adam(const nnReal _eta, const nnReal beta1, const nnReal beta2,
    const nnReal betat1, const nnReal betat2, Saru& _gen) :
  eta(_eta*std::sqrt(1-betat2)/(1-betat1)), B1(beta1), B2(beta2) {}

  #pragma omp declare simd notinbranch simdlen(VEC_WIDTH)
  inline nnReal step(const nnReal&grad, nnReal&M1, nnReal&M2, const nnReal fac){
    const nnReal DW  = grad * fac;
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = B2 * M2 + (1-B2) * DW*DW;
    M2 = M2<M1*M1? M1*M1 : M2; // gradient clipping to 1
    //const nnReal _M2 = std::sqrt(M2 + nnEPS);
    //return eta*M1/_M2;
    return eta*M1/(nnEPS + std::sqrt(M2));
    //return eta*(B1*M1 + (1-B1)*DW)/_M2;
  }
};

struct Momentum {
  const nnReal eta, B1;
  Momentum(const nnReal _eta, const nnReal beta1, const nnReal beta2,
    const nnReal betat1, const nnReal betat2, Saru& _gen) :
  eta(_eta), B1(beta1) {}

  inline nnReal step(const nnReal&grad, nnReal&M1, nnReal&M2, const nnReal fac){
    const nnReal DW  = grad * fac;
    M1 = B1 * M1 + eta * DW;
    return M1;
  }
};

struct SGD {
  const nnReal eta;
  SGD(const nnReal _eta, const nnReal beta1, const nnReal beta2,
    const nnReal betat1, const nnReal betat2, Saru& _gen) :
  eta(_eta) {}

  inline nnReal step(const nnReal&grad, nnReal&M1, nnReal&M2, const nnReal fac){
    const nnReal DW  = grad * fac;
    return eta*DW;
  }
};

class Optimizer
{
 protected:
  const MPI_Comm mastersComm;
  const Uint learn_size;
  const Real eta, beta_1, beta_2;
  long unsigned nStep = 0;
  Real beta_t_1 = beta_1, beta_t_2 = beta_2;
  const Real lambda, epsAnneal, tgtUpdateAlpha;
  const Parameters * const weights;
  const Parameters * const tgt_weights;
  const Parameters * const gradSum;
  const Parameters * const _1stMom;
  const Parameters * const _2ndMom;
  vector<std::mt19937>& generators;
  Uint cntUpdateDelay = 0, totGrads = 0;
  MPI_Request paramRequest = MPI_REQUEST_NULL;
  MPI_Request batchRequest = MPI_REQUEST_NULL;
  //const Real alpha_eSGD, gamma_eSGD, eta_eSGD, eps_eSGD, delay;
  //const Uint L_eSGD;
  //nnReal *const _muW_eSGD, *const _muB_eSGD;

 public:
  Optimizer(Settings&S, const Parameters*const W, const Parameters*const W_TGT,
    const Real B1=.9, const Real B2=.999) : mastersComm(S.mastersComm),
    learn_size(S.learner_size), eta(S.learnrate), beta_1(B1), beta_2(B2),
    lambda(S.nnLambda), epsAnneal(S.epsAnneal), tgtUpdateAlpha(S.targetDelay),
    weights(W), tgt_weights(W_TGT), gradSum(W->allocateGrad()),
    _1stMom(W->allocateGrad()), _2ndMom(W->allocateGrad()),
    generators(S.generators) {  }
  //alpha_eSGD(0.75), gamma_eSGD(10.), eta_eSGD(.1/_s.targetDelay),
  //eps_eSGD(1e-3), delay(_s.targetDelay), L_eSGD(_s.targetDelay),
  //_muW_eSGD(initClean(nWeights)), _muB_eSGD(initClean(nBiases))

  virtual ~Optimizer() {
   _dispose_object(gradSum); _dispose_object(_1stMom); _dispose_object(_2ndMom);
  }

  inline void prepare_update(const int batchsize, const vector<Parameters*>& grads) {
    prepare_update(batchsize, &grads);
  }

  void prepare_update(const int batchsize, const vector<Parameters*>* grads = nullptr)
  {
    totGrads = batchsize;
    if(grads not_eq nullptr) //add up gradients across threads
      gradSum->reduceThreadsGrad(*grads);

    if (learn_size > 1) { //add up gradients across master ranks
      MPI_Iallreduce(MPI_IN_PLACE, gradSum->params, gradSum->nParams, MPI_NNVALUE_TYPE, MPI_SUM, mastersComm, &paramRequest);
      MPI_Iallreduce(MPI_IN_PLACE, &totGrads, 1, MPI_UNSIGNED, MPI_SUM, mastersComm, &batchRequest);
    }
    nStep++;
  }

  void apply_update()
  {
    if(nStep == 0) die("nStep == 0");
    if(learn_size > 1) {
      if(batchRequest == MPI_REQUEST_NULL)
        die("I am in finalize without having started a reduction");
      if(paramRequest == MPI_REQUEST_NULL)
        die("I am in finalize without having started a reduction");
      MPI_Wait(&paramRequest, MPI_STATUS_IGNORE);
      MPI_Wait(&batchRequest, MPI_STATUS_IGNORE);
    }
    using Algorithm = Adam;
    //update is deterministic: can be handled independently by each node
    //communication overhead is probably greater than a parallelised sum
    assert(totGrads>0);
    const Real factor = 1./totGrads;
    nnReal* const paramAry = weights->params;    
    assert(eta < 2e-3); //super upper bound for NN, srsly
    const Real _eta =eta +1e-3*std::max(5-std::log10((Real)nStep),(Real)0)/5;

    #pragma omp parallel
    {
      const Uint thrID = static_cast<Uint>(omp_get_thread_num());
      Saru gen(nStep, thrID, generators[thrID]());
      Algorithm algo(_eta, beta_1, beta_2, beta_t_1, beta_t_2, gen);
      nnReal* const mom1 = _1stMom->params;    
      nnReal* const mom2 = _2ndMom->params;    
      nnReal* const grad = gradSum->params;    

      #pragma omp for simd aligned(paramAry, mom1, mom2, grad : VEC_WIDTH)
      for (Uint i=0; i<weights->nParams; i++)
        paramAry[i] += algo.step(grad[i], mom1[i], mom2[i], factor);
    }

    gradSum->clear();
    // Needed by Adam optimization algorithm:
    beta_t_1 *= beta_1;
    if (beta_t_1<nnEPS) beta_t_1 = 0;
    beta_t_2 *= beta_2;
    if (beta_t_2<nnEPS) beta_t_2 = 0;

    if(lambda>nnEPS) weights->penalization(lambda);

    // update frozen weights:
    if(tgtUpdateAlpha > 0 && tgt_weights not_eq nullptr) {
      if (cntUpdateDelay == 0) {
        cntUpdateDelay = tgtUpdateAlpha;

        if(tgtUpdateAlpha>=1) tgt_weights->copy(weights);
        else {
          nnReal* const targetAry = tgt_weights->params;
          #pragma omp parallel for simd aligned(paramAry, targetAry : VEC_WIDTH) 
          for(Uint j=0; j<weights->nParams; j++)
            targetAry[j] += tgtUpdateAlpha*(paramAry[j] - targetAry[j]);
        }
      }
      if(cntUpdateDelay>0) cntUpdateDelay--;
    }
  }

  void save(const string fname)
  {
    weights->save(fname+"_weights");
    if(tgt_weights not_eq nullptr) tgt_weights->save(fname+"_tgt_weights");
    _1stMom->save(fname+"_1stMom");
    _2ndMom->save(fname+"_2ndMom");

    if(nStep % 100000 == 0 && nStep > 0) {
      ostringstream ss; ss << std::setw(9) << std::setfill('0') << nStep;
      weights->save(fname+"_"+ss.str()+"_weights");
      _1stMom->save(fname+"_"+ss.str()+"_1stMom" );
      _2ndMom->save(fname+"_"+ss.str()+"_2ndMom" );
    }
  }
  int restart(const string fname)
  {
    int ret = 0;
    ret = weights->restart(fname+"_weights");
    if(tgt_weights not_eq nullptr) {
      int missing_tgt = tgt_weights->restart(fname+"_tgt_weights");
      if (missing_tgt) tgt_weights->copy(weights);
    }
    _1stMom->restart(fname+"_1stMom");
    _2ndMom->restart(fname+"_2ndMom");
    return ret;
  }
};

#if 0
void save_recurrent_connections(const string fname)
{
  const Uint nNeurons(net->getnNeurons());
  const Uint nAgents(net->getnAgents()), nStates(net->getnStates());
  string nameBackup = fname + "_mems_tmp";
  ofstream out(nameBackup.c_str());
  if (!out.good())
    _die("Unable to open save into file %s\n", nameBackup.c_str());

  for(Uint agentID=0; agentID<nAgents; agentID++) {
    for (Uint j=0; j<nNeurons; j++)
      out << net->mem[agentID]->outvals[j] << "\n";
    for (Uint j=0; j<nStates;  j++)
      out << net->mem[agentID]->ostates[j] << "\n";
  }
  out.flush();
  out.close();
  string command = "cp " + nameBackup + " " + fname + "_mems";
  system(command.c_str());
}

bool restart_recurrent_connections(const string fname)
{
  const Uint nNeurons(net->getnNeurons());
  const Uint nAgents(net->getnAgents()), nStates(net->getnStates());

  string nameBackup = fname + "_mems";
  ifstream in(nameBackup.c_str());
  debugN("Reading from %s", nameBackup.c_str());
  if (!in.good()) {
    error("Couldnt open file %s \n", nameBackup.c_str());
    return false;
  }

  nnReal tmp;
  for(Uint agentID=0; agentID<nAgents; agentID++) {
    for (Uint j=0; j<nNeurons; j++) {
      in >> tmp;
      if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
      net->mem[agentID]->outvals[j] = tmp;
    }
    for (Uint j=0; j<nStates; j++) {
      in >> tmp;
      if (std::isnan(tmp) || std::isinf(tmp)) tmp=0.;
      net->mem[agentID]->ostates[j] = tmp;
    }
  }
  in.close();
  return true;
}
#endif
