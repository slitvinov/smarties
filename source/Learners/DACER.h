/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "../Network/Builder.h"
#include "Learner_offPolicy.h"

#include "../Math/Gaussian_mixture.h"

#ifndef NEXPERTS
#define NEXPERTS 1
#warning "Using Mixture_advantage with 1 expert"
#endif

#include "../Math/Discrete_policy.h"
#include "DACER.cpp"
//#define DACER_simpleSigma

class DACER_disc : public DACER<Discrete_policy, Uint>
{
  static vector<Uint> count_outputs(const ActionInfo*const aI) {
    return vector<Uint>{1, aI->maxLabel};
  }
  static vector<Uint> count_pol_starts(const ActionInfo*const aI) {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[1]};
  }

 public:
  static Uint getnOutputs(const ActionInfo*const aI) {
    return 1 + aI->maxLabel;
  }
  static Uint getnDimPolicy(const ActionInfo*const aI) {
    return aI->maxLabel;
  }

 public:
  DACER_disc(Environment*const _env, Settings& _set) :
  DACER(_env, _set, count_outputs(&_env->aI), count_pol_starts(&_env->aI) )
  {
    printf("Discrete-action DACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(), print(net_outputs).c_str());
    F.push_back(new Approximator("net", _set, input, data));
    vector<Uint> nouts{1, nA};
    Builder build = F[0]->buildFromSettings(_set, nouts);
    F[0]->initializeNetwork(build);
  }
};

class DACER_experts : public DACER<Gaussian_mixture<NEXPERTS>, Rvec>
{
  static vector<Uint> count_outputs(const ActionInfo& aI) {
    return vector<Uint>{1, NEXPERTS, NEXPERTS*aI.dim, NEXPERTS*aI.dim};
  }
  static vector<Uint> count_pol_starts(const ActionInfo& aI) {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[1], indices[2], indices[3]};
  }
 public:
  static Uint getnOutputs(const ActionInfo*const aI) {
    return 1 + getnDimPolicy(aI);
  }
  static Uint getnDimPolicy(const ActionInfo*const aI) {
    return NEXPERTS*(1 +2*aI->dim);
  }

  DACER_experts(Environment*const _env, Settings& _set) :
  DACER(_env, _set, count_outputs(_env->aI), count_pol_starts(_env->aI) )
  {
    printf("Mixture-of-experts DACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(), print(net_outputs).c_str());

    F.push_back(new Approximator("net", _set, input, data));

    vector<Uint> nouts{1, NEXPERTS, NEXPERTS * nA};
    #ifndef DACER_simpleSigma // network outputs also sigmas
      nouts.push_back(NEXPERTS * nA);
    #endif

    Builder build = F[0]->buildFromSettings(_set, nouts);

    Rvec initBias;
    initBias.push_back(0); // state value
    Gaussian_mixture<NEXPERTS>::setInitial_noStdev(&aInfo, initBias);

    #ifdef DACER_simpleSigma // sigma not linked to network: parametric output
      build.setLastLayersBias(initBias);
      Real initParam = Gaussian_mixture<NEXPERTS>::precision_inverse(explNoise);
      build.addParamLayer(NEXPERTS * nA, "Linear", initParam);
    #else
      Gaussian_mixture<NEXPERTS>::setInitial_Stdev(&aInfo, initBias, explNoise);
      build.setLastLayersBias(initBias);
    #endif
    F[0]->initializeNetwork(build, STD_GRADCUT);

    {  // TEST FINITE DIFFERENCES:
      Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
      std::normal_distribution<Real> dist(0, 1);
      for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
      for(Uint i=0; i<mu.size(); i++) mu[i] = dist(generators[0]);
      Real norm = 0;
      for(Uint i=0; i<NEXPERTS; i++) {
        mu[i] = std::exp(mu[i]);
        norm += mu[i];
      }
      for(Uint i=0; i<NEXPERTS; i++) mu[i] = mu[i]/norm;
      for(Uint i=NEXPERTS*(1+nA);i<NEXPERTS*(1+2*nA);i++) mu[i]=std::exp(mu[i]);

      Gaussian_mixture<NEXPERTS>  pol = prepare_policy(output);
      Rvec act = pol.finalize(1, &generators[0], mu);
      pol.prepare(act, mu);
      pol.test(act, mu);
    }
  }
};

#if 0
class DACER_cont : public DACER<Quadratic_advantage, Gaussian_policy, Rvec >
{
  static vector<Uint> count_outputs(const ActionInfo& aI)
  {
    const Uint nL = Quadratic_term::compute_nL(&aI);
    #if defined ACER_RELAX
      return vector<Uint>{1, nL, aI.dim, aI.dim};
    #else
      return vector<Uint>{1, nL, aI.dim, aI.dim, aI.dim};
    #endif
  }
  static vector<Uint> count_pol_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    #if defined ACER_RELAX
    return vector<Uint>{indices[2], indices[3]};
    #else
    return vector<Uint>{indices[2], indices[4]};
    #endif
  }
  static vector<Uint> count_adv_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    #if defined ACER_RELAX
    return vector<Uint>{indices[1]};
    #else
    return vector<Uint>{indices[1], indices[3]};
    #endif
  }

 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    const Uint nL = Quadratic_advantage::compute_nL(aI);
    #if defined ACER_RELAX // I output V(s), P(s), pol(s), prec(s) (and variate)
      return 1 + nL + aI->dim + aI->dim;
    #else // I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
      return 1 + nL + aI->dim + aI->dim + aI->dim;
    #endif
  }
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim;
  }

  DACER_cont(Environment*const _env, Settings& _set) :
  DACER(_env, _set, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Continuous-action DACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", _set, input, data));
    vector<Uint> nouts{1, nL, nA};
    #ifndef DACER_simpleSigma
      nouts.push_back(nA);
    #endif
    Builder build = F[0]->buildFromSettings(_set, nouts);
    #ifdef DACER_simpleSigma
      build.addParamLayer(nA, "Linear", -2*std::log(explNoise));
    #endif
    F[0]->initializeNetwork(build);
  }
};
#endif
