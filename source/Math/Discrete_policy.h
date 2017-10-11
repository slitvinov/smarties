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
  const vector<Real>& netOutputs;
  const vector<Real> unnorm;
  const Real normalization;
  const vector<Real> probs;

  Uint sampAct;
  Real sampLogPonPolicy=0, sampLogPBehavior=0, sampImpWeight=0;

  Discrete_policy(const vector <Uint>& start, const ActionInfo*const aI,
    const vector<Real>& out) : aInfo(aI), start_prob(start[0]), nA(aI->maxLabel), netOutputs(out), unnorm(extract_unnorm()),
    normalization(compute_norm()), probs(extract_probabilities())
    {
      //printf("Discrete_policy: %u %u %u %lu %lu %lu %lu\n",
      //start_prob,start_vals,nA,netOutputs.size(),
      //unnorm.size(),vals.size(),probs.size());
    }

 private:
  inline vector<Real> extract_unnorm() const
  {
    assert(netOutputs.size()>=start_prob+nA);
    vector<Real> ret(nA);
    for (Uint j=0; j<nA; j++) ret[j] = prob_func(netOutputs[start_prob+j]);
    return ret;
  }

  inline Real compute_norm() const
  {
    assert(unnorm.size()==nA);
    Real ret = 0;
    for (Uint j=0; j<nA; j++) { ret += unnorm[j]; assert(unnorm[j]>0); }
    return ret + std::numeric_limits<Real>::epsilon();
  }

  inline vector<Real> extract_probabilities() const
  {
    assert(unnorm.size()==nA);
    vector<Real> ret(nA);
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
  inline void prepare(const vector<Real>& unbact, const vector<Real>& beta, const bool bGeometric, const Discrete_policy*const pol_hat = nullptr)
  {
    sampAct = map_action(unbact);
    sampLogPonPolicy = evalLogProbability(sampAct);
    sampLogPBehavior = evalBehavior(sampAct, beta);
    const Real logW = sampLogPonPolicy - sampLogPBehavior;
    sampImpWeight = bGeometric ? safeExp(logW/nA) : safeExp(logW);
    sampImpWeight = std::min(MAX_IMPW, sampImpWeight);
  }

  void test(const Uint act, const Discrete_policy*const pol_hat) const;

  static inline Real evalBehavior(const Uint& act, const vector<Real>& beta)
  {
    return std::log(beta[act]);
  }

  inline Uint sample(mt19937*const gen, const vector<Real>& beta) const
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
  inline vector<Real> control_grad(const Advantage_t*const adv, const Real eta) const
  {
    vector<Real> ret(nA, 0);
    for (Uint j=0; j<nA; j++)
      ret[j] = eta*adv->computeAdvantage(j)/normalization;
    return ret;
  }

  inline vector<Real> policy_grad(const Uint act, const Real factor) const
  {
    vector<Real> ret(nA);
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
  inline Real kl_divergence_opp(const vector<Real>& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++)
      ret += probs[i]*std::log(probs[i]/beta[i]);
    return ret;
  }

  inline vector<Real> div_kl_grad(const Discrete_policy*const pol_hat, const Real fac = 1) const
  {
    vector<Real> ret(nA, 0);
    for (Uint j=0; j<nA; j++)
      ret[j] = fac*(1./normalization - pol_hat->probs[j]/unnorm[j]);
    return ret;
  }
  inline vector<Real> div_kl_opp_grad(const Discrete_policy*const pol_hat, const Real fac = 1) const
  {
    vector<Real> ret(nA, 0);
    for (Uint j=0; j<nA; j++){
      const Real tmp=fac*(1+std::log(probs[j]/pol_hat->probs[j]))/normalization;
      for (Uint i=0; i<nA; i++) ret[i] += tmp*((i==j)-probs[j]);
    }
    return ret;
  }
  inline vector<Real> div_kl_opp_grad(const vector<Real>& beta, const Real fac = 1) const
  {
    vector<Real> ret(nA, 0);
    for (Uint j=0; j<nA; j++){
      const Real tmp = fac*(1+std::log(probs[j]/beta[j]))/normalization;
      for (Uint i=0; i<nA; i++) ret[i] += tmp*((i==j)-probs[j]);
    }
    return ret;
  }

  inline void finalize_grad(const vector<Real>&grad, vector<Real>&netGradient) const
  {
    assert(netGradient.size()>=start_prob+nA && grad.size() == nA);
    for (Uint j=0; j<nA; j++)
      netGradient[start_prob+j] = grad[j]*prob_diff(netOutputs[start_prob+j]);
  }

  inline vector<Real> getProbs() const
  {
    return probs;
  }
  inline vector<Real> getBeta() const
  {
    return probs;
  }

  inline Uint finalize(const Real bSample, mt19937*const gen, const vector<Real>& beta) const
  {
    if(bSample) return sample(gen, beta);
    else return //the index of max Q:
      std::distance(probs.begin(), std::max_element(probs.begin(),probs.end()));
  }

  static inline void anneal_beta(vector<Real>& beta, const Real eps)
  {
    for(Uint i=0;i<beta.size();i++) beta[i] = (1-eps)*beta[i] +eps/beta.size();
  }

  inline Uint map_action(const vector<Real>& sent) const
  {
    return aInfo->actionToLabel(sent);
  }
  static inline Uint compute_nA(const ActionInfo* const aI)
  {
    assert(aI->maxLabel);
    return aI->maxLabel;
  }
};
/*
 inline Real diagTerm(const vector<Real>& S, const vector<Real>& mu,
      const vector<Real>& a) const
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
   const vector<Real>& netOutputs;
   const vector<Real> advantages;
   const Discrete_policy* const policy;

   static inline Uint compute_nL(const ActionInfo* const aI)
   {
     assert(aI->maxLabel);
     return aI->maxLabel;
   }

   Discrete_advantage(const vector<Uint>& starts, const ActionInfo* const aI,
     const vector<Real>& out, const Discrete_policy*const pol = nullptr) : aInfo(aI), start_adv(starts[0]), nA(aI->maxLabel), netOutputs(out),
     advantages(extract(out)), policy(pol) { assert(starts.size()==1); }

  protected:
   inline vector<Real> extract(const vector<Real> & v) const
   {
     assert(v.size() >= start_adv + nA);
     return vector<Real>( &(v[start_adv]), &(v[start_adv +nA]) );
   }
   inline Real expectedAdvantage() const
   {
     Real ret = 0;
     for (Uint j=0; j<nA; j++) ret += policy->probs[j]*advantages[j];
     return ret;
   }

  public:
   inline void grad(const Uint act, const Real Qer, vector<Real>&netGradient) const
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
