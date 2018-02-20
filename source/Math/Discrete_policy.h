/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Utils.h"

struct Discrete_policy
{
  const ActionInfo* const aInfo;
  const Uint start_prob, nA;
  const Rvec netOutputs;
  const Rvec unnorm;
  const Real normalization;
  const Rvec probs;

  Uint sampAct;
  Real sampLogPonPolicy=0, sampLogPBehavior=0, sampImpWeight=0, sampRhoWeight=0;

  Discrete_policy(const vector<Uint>& start, const ActionInfo*const aI,
    const Rvec& out) : aInfo(aI), start_prob(start[0]), nA(aI->maxLabel), netOutputs(out), unnorm(extract_unnorm()),
    normalization(compute_norm()), probs(extract_probabilities())
    {
      //printf("Discrete_policy: %u %u %u %lu %lu %lu %lu\n",
      //start_prob,start_vals,nA,netOutputs.size(),
      //unnorm.size(),vals.size(),probs.size());
    }

 private:
  inline Rvec extract_unnorm() const
  {
    assert(netOutputs.size()>=start_prob+nA);
    Rvec ret(nA);
    for (Uint j=0; j<nA; j++) ret[j] = prob_func(netOutputs[start_prob+j]);
    return ret;
  }

  inline Real compute_norm() const
  {
    assert(unnorm.size()==nA);
    Real ret = 0;
    for (Uint j=0; j<nA; j++) { ret += unnorm[j]; assert(unnorm[j]>0); }
    return ret + numeric_limits<Real>::epsilon();
  }

  inline Rvec extract_probabilities() const
  {
    assert(unnorm.size()==nA);
    Rvec ret(nA);
    for (Uint j=0; j<nA; j++) ret[j] = unnorm[j]/normalization;
    return ret;
  }

  static inline Real prob_func(const Real val)
  {
    //return safeExp(val) + ACER_MIN_PROB;
    return 0.5*(val + std::sqrt(val*val+1)) + ACER_MIN_PROB;
  }

  static inline Real prob_diff(const Real val)
  {
    //return safeExp(val);
    return 0.5*(1.+val/std::sqrt(val*val+1));
  }

 public:
  inline void prepare(const Rvec& unbact, const Rvec& beta)
  {
    sampAct = map_action(unbact);
    sampLogPonPolicy = evalLogProbability(sampAct);
    sampLogPBehavior = evalBehavior(sampAct, beta);
    const Real logW = sampLogPonPolicy - sampLogPBehavior;
    sampImpWeight = safeExp(logW);
    sampRhoWeight = sampImpWeight;
  }

  void test(const Uint act, const Discrete_policy*const pol_hat) const;

  static inline Real evalBehavior(const Uint& act, const Rvec& beta)
  {
    return std::log(beta[act]);
  }

  static inline Uint sample(mt19937*const gen, const Rvec& beta)
  {
    std::discrete_distribution<Uint> dist(beta.begin(), beta.end());
    return dist(*gen);
  }

  inline Uint sample(mt19937*const gen) const
  {
    std::discrete_distribution<Uint> dist(probs.begin(), probs.end());
    return dist(*gen);
  }

  inline Real evalLogProbability(const Uint act) const
  {
    assert(act<=nA && probs.size()==nA);
    return std::log(probs[act]);
  }

  template<typename Advantage_t>
  inline Rvec control_grad(const Advantage_t*const adv, const Real eta) const
  {
    Rvec ret(nA, 0);
    for (Uint j=0; j<nA; j++)
      ret[j] = eta*adv->computeAdvantage(j)/normalization;
    return ret;
  }

  inline Rvec policy_grad(const Uint act, const Real factor) const
  {
    Rvec ret(nA);
    //for (Uint i=0; i<nA; i++) ret[i] = factor*(((i==act) ? 1 : 0) -probs[i]);
    for (Uint i=0; i<nA; i++) ret[i] = -factor/normalization;
    ret[act] += factor/unnorm[act];
    return ret;
  }

  inline Rvec policy_grad(const Uint act, const Rvec& beta, const Real factor) const
  {
    Rvec ret(nA);
    abort(); // what to do here?
    //for (Uint i=0; i<nA; i++) ret[i] = factor*(((i==act) ? 1 : 0) -probs[i]);
    for (Uint i=0; i<nA; i++) ret[i] = -factor/normalization;
    ret[act] += factor/unnorm[act];
    return ret;
  }

  inline Real kl_divergence(const Discrete_policy*const pol_hat) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++)
      ret += pol_hat->probs[i]*(std::log(pol_hat->probs[i]/probs[i]));
    return ret;
  }
  inline Real kl_divergence_opp(const Discrete_policy*const pol_hat) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++)
      ret += probs[i]*std::log(probs[i]/pol_hat->probs[i]);
    return ret;
  }
  inline Real kl_divergence_opp(const Rvec& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++)
      ret += probs[i]*std::log(probs[i]/beta[i]);
    return ret;
  }

  inline Rvec div_kl_grad(const Discrete_policy*const pol_hat, const Real fac = 1) const
  {
    Rvec ret(nA, 0);
    for (Uint j=0; j<nA; j++)
      ret[j] = fac*(1./normalization - pol_hat->probs[j]/unnorm[j]);
    return ret;
  }
  inline Rvec div_kl_opp_grad(const Discrete_policy*const pol_hat, const Real fac = 1) const
  {
    Rvec ret(nA, 0);
    for (Uint j=0; j<nA; j++){
      const Real tmp=fac*(1+std::log(probs[j]/pol_hat->probs[j]))/normalization;
      for (Uint i=0; i<nA; i++) ret[i] += tmp*((i==j)-probs[j]);
    }
    return ret;
  }
  inline Rvec div_kl_opp_grad(const Rvec& beta, const Real fac = 1) const
  {
    Rvec ret(nA, 0);
    for (Uint j=0; j<nA; j++){
      const Real tmp = fac*(1+std::log(probs[j]/beta[j]))/normalization;
      for (Uint i=0; i<nA; i++) ret[i] += tmp*((i==j)-probs[j]);
    }
    return ret;
  }

  inline void finalize_grad(const Rvec&grad, Rvec&netGradient) const
  {
    assert(netGradient.size()>=start_prob+nA && grad.size() == nA);
    for (Uint j=0; j<nA; j++)
      netGradient[start_prob+j] = grad[j]*prob_diff(netOutputs[start_prob+j]);
  }

  inline Rvec getProbs() const
  {
    return probs;
  }
  inline Rvec getVector() const
  {
    return probs;
  }

  inline Uint finalize(const bool bSample, mt19937*const gen, const Rvec& beta)
  {
    sampAct = bSample? sample(gen, beta) :
      std::distance(probs.begin(), std::max_element(probs.begin(),probs.end()));
    return sampAct; //the index of max Q
  }

  static inline void anneal_beta(Rvec& beta, const Real eps)
  {
    for(Uint i=0;i<beta.size();i++) beta[i] = (1-eps)*beta[i] +eps/beta.size();
  }

  inline Uint map_action(const Rvec& sent) const
  {
    return aInfo->actionToLabel(sent);
  }
  static inline Uint compute_nA(const ActionInfo* const aI)
  {
    assert(aI->maxLabel);
    return aI->maxLabel;
  }
  Uint updateOrUhState(Rvec& state, Rvec& beta,
    const Uint act, const Real step) {
    // not applicable to discrete action spaces
    return act;
  }
};
/*
 inline Real diagTerm(const Rvec& S, const Rvec& mu,
      const Rvec& a) const
  {
    assert(S.size() == nA);
    assert(a.size() == nA);
    assert(mu.size() == nA);
    Real Q = 0;
    for (Uint j=0; j<nA; j++) Q += S[j]*std::pow(mu[j]-a[j],2);
    return Q;
  }
 */

 struct Discrete_advantage
 {
   const ActionInfo* const aInfo;
   const Uint start_adv, nA;
   const Rvec& netOutputs;
   const Rvec advantages;
   const Discrete_policy* const policy;

   static inline Uint compute_nL(const ActionInfo* const aI)
   {
     assert(aI->maxLabel);
     return aI->maxLabel;
   }

   Discrete_advantage(const vector<Uint>& starts, const ActionInfo* const aI,
     const Rvec& out, const Discrete_policy*const pol = nullptr) : aInfo(aI), start_adv(starts[0]), nA(aI->maxLabel), netOutputs(out),
     advantages(extract(out)), policy(pol) {}

  protected:
   inline Rvec extract(const Rvec & v) const
   {
     assert(v.size() >= start_adv + nA);
     return Rvec( &(v[start_adv]), &(v[start_adv +nA]) );
   }
   inline Real expectedAdvantage() const
   {
     Real ret = 0;
     for (Uint j=0; j<nA; j++) ret += policy->probs[j]*advantages[j];
     return ret;
   }

  public:
   inline void grad(const Uint act, const Real Qer, Rvec&netGradient) const
   {
     if(policy not_eq nullptr)
       for (Uint j=0; j<nA; j++)
         netGradient[start_adv+j] = Qer*((j==act ? 1 : 0) - policy->probs[j]);
     else
       for (Uint j=0; j<nA; j++)
         netGradient[start_adv+j] = Qer* (j==act ? 1 : 0);
   }

   Real computeAdvantage(const Uint action) const
   {
     if(policy not_eq nullptr)
       return advantages[action]-expectedAdvantage(); //subtract expectation from advantage of action
     else return advantages[action];
   }

   Real computeAdvantageNoncentral(const Uint action) const
   {
     return advantages[action];
   }

   Rvec getParam() const {
     return advantages;
   }

   inline Real advantageVariance() const
   {
     assert(policy not_eq nullptr);
     if(policy == nullptr) return 0;
     const Real base = expectedAdvantage();
     Real ret = 0;
     for (Uint j=0; j<nA; j++)
       ret += policy->probs[j]*(advantages[j]-base)*(advantages[j]-base);
     return ret;
   }

   void test(const Uint& act, mt19937*const gen) const {}
 };
