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
#include "../Math/Mixture_advantage_prova.h"
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
      return vector<Uint>{1, nL, aI.dim, aI.dim, 2};
    #else
      return vector<Uint>{1, nL, aI.dim, aI.dim, aI.dim, 2};
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
      return 1 + nL + aI->dim + aI->dim + 2;
    #else // I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
      return 1 + nL + aI->dim + aI->dim + aI->dim + 2;
    #endif
  }
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim;
  }

  RACER_cont(Environment*const _env, Settings& settings) :
  RACER(_env, settings, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Continuous-action RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", settings, input, data));
    vector<Uint> nouts{1, nL, nA};
    #ifndef simpleSigma
      nouts.push_back(nA);
    #endif
    Builder build = F[0]->buildFromSettings(settings, nouts);
    #ifdef simpleSigma
      build.addParamLayer(nA, "Linear", -2*std::log(greedyEps));
    #endif
    //add klDiv penalty coefficient layer, and stdv of Q distribution
    build.addParamLayer(2, "Exp", 1/settings.klDivConstraint);
    F[0]->initializeNetwork(build);
  }
};

class RACER_disc : public RACER<Discrete_advantage, Discrete_policy, Uint>
{
  static vector<Uint> count_outputs(const ActionInfo*const aI)
  {
    return vector<Uint>{1, aI->maxLabel, aI->maxLabel, 2};
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
    return 1 + aI->maxLabel + aI->maxLabel + 2;
  }
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return aI->maxLabel;
  }

 public:
  RACER_disc(Environment*const _env, Settings& settings) :
  RACER(_env, settings, count_outputs(&_env->aI), count_pol_starts(&_env->aI), count_adv_starts(&_env->aI) )
  {
    printf("Discrete-action RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", settings, input, data));
    vector<Uint> nouts{1, nL, nA};
    Builder build = F[0]->buildFromSettings(settings, nouts);
    //add klDiv penalty coefficient layer, and stdv of Q distribution
    build.addParamLayer(2, "Exp", 1);
    F[0]->initializeNetwork(build);
  }
};

class RACER_experts : public RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, vector<Real>>
{
  static vector<Uint> count_outputs(const ActionInfo& aI)
  {
    const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(&aI);
    return vector<Uint>{1, nL, NEXPERTS, NEXPERTS*aI.dim, NEXPERTS*aI.dim, 2};
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
    return 1 + nL + NEXPERTS*(1 +2*aI->dim) + 2;
  }
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return NEXPERTS*(1 +2*aI->dim);
  }

  RACER_experts(Environment*const _env, Settings& settings) :
  RACER(_env, settings, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Mixture-of-experts RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", settings, input, data));
    vector<Uint> nouts{1, nL, NEXPERTS, NEXPERTS * nA};
    #ifndef simpleSigma
      nouts.push_back(NEXPERTS * nA);
    #endif
    Builder build = F[0]->buildFromSettings(settings, nouts);
    #ifdef simpleSigma
      build.addParamLayer(NEXPERTS * nA, "Linear", -2*std::log(greedyEps));
    #endif
    //add klDiv penalty coefficient layer, and stdv of Q distribution
    build.addParamLayer(2, "Exp", {1/settings.klDivConstraint, 1});

    F[0]->initializeNetwork(build);
  }
};
