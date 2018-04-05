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

struct Adam {
  const nnReal eta, B1, B2, lambda, fac;
  Adam(nnReal _eta, nnReal beta1, nnReal beta2, nnReal betat1,
    nnReal betat2, nnReal _lambda, nnReal _fac, Saru& _gen) :
    eta(_eta*std::sqrt(1-betat2)/(1-betat1)), B1(beta1), B2(beta2),
    lambda(_lambda), fac(_fac) {}

  #pragma omp declare simd notinbranch //simdlen(VEC_WIDTH)
  inline nnReal step(const nnReal grad,nnReal&M1,nnReal&M2,const nnReal W) const
  {
    #ifdef NET_L1_PENAL
      const nnReal penal = -(W>0 ? lambda : -lambda);
    #else
      const nnReal penal = - W*lambda;
    #endif
    const nnReal DW = fac * grad;
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = B2 * M2 + (1-B2) * DW*DW;
    #ifdef NESTEROV_ADAM
      const nnReal numer = B1*M1 + (1-B1)*DW;
    #else
      const nnReal numer = M1;
    #endif
    #ifdef SAFE_ADAM
      // prevent rare gradient blow ups. allows dW<= eta*sqrt(10). if pre update
      // M2 and M1 were both 0, this is what normally happens with Adam
      // Actually I can't think of a situation where, except due to finite
      // precision, this next like will not be reduntant...
      M2 = M2 < M1*M1/10 ? M1*M1/10 : M2;
      const nnReal ret =  eta * (numer /  std::sqrt(nnEPS + M2)  +penal);
      //return eta * (numer / (nnEPS + std::sqrt(M2)) +penal);
    #else
      const nnReal ret =  eta * (numer /  std::sqrt(nnEPS + M2)  +penal);
    #endif
    assert(not std::isnan(ret) && not std::isinf(ret));
    return ret;
  }
};

struct AdaMax {
  const nnReal eta, B1, B2, lambda, fac;
  AdaMax(nnReal _eta, nnReal beta1, nnReal beta2, nnReal betat1,
    nnReal betat2, nnReal _lambda, nnReal _fac, Saru& _gen) :
    eta(_eta), B1(beta1), B2(beta2), lambda(_lambda), fac(_fac) {}

  #pragma omp declare simd notinbranch //simdlen(VEC_WIDTH)
  inline nnReal step(const nnReal grad,nnReal&M1,nnReal&M2,const nnReal W) const
  {
    #ifdef NET_L1_PENAL
      const nnReal DW = grad * fac -(W>0 ? lambda : -lambda);
    #else
      const nnReal DW = grad * fac - W*lambda;
    #endif
    M1 = B1 * M1 + (1-B1) * DW;
    M2 = std::max(B2 * M2, std::fabs(DW));
    #ifdef NESTEROV_ADAM
      return eta * (B1*M1 + (1-B1)*DW)/std::max(M2, nnEPS);
    #else
      return eta * M1                 /std::max(M2, nnEPS);
    #endif
  }
};

template <class T>
struct Entropy : public T {
  Saru& gen;
  const nnReal eps;
  Entropy(nnReal _eta, nnReal beta1, nnReal beta2, nnReal bett1, nnReal bett2,
    nnReal _lambda, nnReal _fac, Saru& _gen) : T(_eta, beta1, beta2, bett1,
    bett2, _lambda, _fac, _gen), gen(_gen), eps(_eta*_lambda) {}

  inline nnReal step(const nnReal grad, nnReal&M1, nnReal&M2, const nnReal W) {
    return T::step(grad, M1, M2, W) + eps * gen.d_mean0_var1();
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
    generators(S.generators) {
      _2ndMom->set(std::sqrt(nnEPS));
      //_2ndMom->set(1);
    }
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
    #ifndef __EntropySGD
      using Algorithm = Adam;
    #else
      using Algorithm = Entropy<Adam>;
    #endif
    //update is deterministic: can be handled independently by each node
    //communication overhead is probably greater than a parallelised sum

    const Real factor = 1./totGrads;
    nnReal* const paramAry = weights->params;
    assert(eta < 2e-3); //super upper bound for NN, srsly
    #ifdef ANNEAL_LEARNR
      const Real _eta = eta / (1 + nStep * ANNEAL_RATE);
    #else
      const Real _eta = eta;
    #endif

    if(totGrads>0) {
      #pragma omp parallel
      {
        const Uint thrID = static_cast<Uint>(omp_get_thread_num());
        Saru gen(nStep, thrID, generators[thrID]()); //needs 3 seeds
        Algorithm algo(_eta,beta_1,beta_2,beta_t_1,beta_t_2,lambda,factor,gen);
        nnReal* const M1 = _1stMom->params;
        nnReal* const M2 = _2ndMom->params;
        nnReal* const G  = gradSum->params;

        #pragma omp for simd aligned(paramAry, M1, M2, G : VEC_WIDTH)
        for (Uint i=0; i<weights->nParams; i++)
        paramAry[i] += algo.step(G[i], M1[i], M2[i], paramAry[i]);
      }
    }
    gradSum->clear();
    // Needed by Adam optimization algorithm:
    beta_t_1 *= beta_1;
    if (beta_t_1<nnEPS) beta_t_1 = 0;
    beta_t_2 *= beta_2;
    if (beta_t_2<nnEPS) beta_t_2 = 0;

    // update frozen weights:
    if(tgtUpdateAlpha > 0 && tgt_weights not_eq nullptr) {
      if (cntUpdateDelay == 0) {
        // the targetDelay setting param can be either >1 or <1.
        // if >1 then it means "every how many steps copy weight to tgt weights"
        cntUpdateDelay = tgtUpdateAlpha;
        if(tgtUpdateAlpha>=1) tgt_weights->copy(weights);
        else { // else is the learning rate of an exponential averaging
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
