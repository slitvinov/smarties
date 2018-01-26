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
#include "../Math/FeatureControlTasks.h"
#include "../Math/Quadratic_advantage.h"

#ifdef ADV_GAUS
#include "../Math/Mixture_advantage_prova2.h"
#warning "Using Mixture_advantage with Gaussian advantages"
#else
#include "../Math/Mixture_advantage.h"
#warning "Using Mixture_advantage with Quadratic advantages"
#endif

#ifndef NEXPERTS
#define NEXPERTS 1
#warning "Using Mixture_advantage with 1 expert"
#endif

#include "../Math/Discrete_policy.h"
#include "RACER.cpp"
//#define simpleSigma

class RACER_cont : public RACER<Quadratic_advantage, Gaussian_policy, vector<Real> >
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

  RACER_cont(Environment*const _env, Settings& _set) :
  RACER(_env, _set, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Continuous-action RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", _set, input, data));
    vector<Uint> nouts{1, nL, nA};
    #ifndef simpleSigma
      nouts.push_back(nA);
    #endif
    Builder build = F[0]->buildFromSettings(_set, nouts);
    #ifdef simpleSigma
      build.addParamLayer(nA, "Linear", -2*std::log(greedyEps));
    #endif
    F[0]->initializeNetwork(build);
  }
};

class RACER_disc : public RACER<Discrete_advantage, Discrete_policy, Uint>
{
  static vector<Uint> count_outputs(const ActionInfo*const aI)
  {
    return vector<Uint>{1, aI->maxLabel, aI->maxLabel};
  }
  static vector<Uint> count_pol_starts(const ActionInfo*const aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[1]};
  }
  static vector<Uint> count_adv_starts(const ActionInfo*const aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[2]};
  }

 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    return 1 + aI->maxLabel + aI->maxLabel;
  }
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return aI->maxLabel;
  }

 public:
  RACER_disc(Environment*const _env, Settings& _set) :
  RACER(_env, _set, count_outputs(&_env->aI), count_pol_starts(&_env->aI), count_adv_starts(&_env->aI) )
  {
    printf("Discrete-action RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", _set, input, data));
    vector<Uint> nouts{1, nL, nA};
    Builder build = F[0]->buildFromSettings(_set, nouts);
    F[0]->initializeNetwork(build);
  }
};

class RACER_experts : public RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, vector<Real>>
{
  static vector<Uint> count_outputs(const ActionInfo& aI)
  {
    const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(&aI);
    return vector<Uint>{1, nL, NEXPERTS, NEXPERTS*aI.dim, NEXPERTS*aI.dim};
  }
  static vector<Uint> count_pol_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[2], indices[3], indices[4]};
  }
  static vector<Uint> count_adv_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[1]};
  }
 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(aI);
    return 1 + nL + NEXPERTS*(1 +2*aI->dim);
  }
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return NEXPERTS*(1 +2*aI->dim);
  }

  RACER_experts(Environment*const _env, Settings& _set) :
  RACER(_env, _set, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Mixture-of-experts RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", _set, input, data));
    vector<Uint> nouts{1, nL, NEXPERTS, NEXPERTS * nA};
    #ifndef simpleSigma
      nouts.push_back(NEXPERTS * nA);
    #endif
    Builder build = F[0]->buildFromSettings(_set, nouts);

    vector<Real> initBias;
    initBias.push_back(0);
    Mixture_advantage<NEXPERTS>::setInitial(&aInfo, initBias);
    Gaussian_mixture<NEXPERTS>::setInitial_noStdev(&aInfo, initBias);

    #ifdef simpleSigma
      build.setLastLayersBias(initBias);
      build.addParamLayer(NEXPERTS * nA, "Linear", std::log(greedyEps));
    #else
      Gaussian_mixture<NEXPERTS>::setInitial_Stdev(&aInfo, initBias, greedyEps);
      build.setLastLayersBias(initBias);
    #endif
    F[0]->initializeNetwork(build);

    {
      vector<Real> output(F[0]->nOutputs()),beta(getnDimPolicy(&aInfo)),act(nA);
      std::normal_distribution<Real> dist(0, 1);
      for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
      for(Uint i=0; i<beta.size(); i++) beta[i] = dist(generators[0]);
      Real norm = 0;
      for(Uint i=0; i<NEXPERTS; i++) {
        beta[i] = std::exp(beta[i]);
        norm += beta[i];
      }
      for(Uint i=0; i<NEXPERTS; i++) beta[i] = beta[i]/norm;
      for(Uint i=NEXPERTS*(1+nA);i<NEXPERTS*(1+2*nA);i++) beta[i]=exp(beta[i]);

      for(Uint i=0; i<act.size(); i++) act[i] = dist(generators[0]);
      act = aInfo.getScaled(act);
      Gaussian_mixture<NEXPERTS>  pol = prepare_policy(output);
      Mixture_advantage<NEXPERTS> adv = prepare_advantage(output, &pol);
      adv.test(act, &generators[0]);
      pol.prepare(act, beta);
      pol.test(act, beta);
    }
  }
};
