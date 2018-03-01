/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Quadratic_term.h"

struct Gaussian_policy
{
  const ActionInfo* const aInfo;
  const Uint start_mean, start_prec, nA;
  const Real acerTrickPow = 1. / std::sqrt(nA);
  const Rvec netOutputs;
  const Rvec mean, stdev, variance, precision;

  Rvec sampAct;
  Real sampLogPonPolicy=0, sampLogPBehavior=0, sampImpWeight=0, sampRhoWeight=0;

  inline Rvec map_action(const Rvec& sent) const {
    return aInfo->getInvScaled(sent);
  }
  static inline Uint compute_nA(const ActionInfo* const aI) {
    assert(aI->dim); return aI->dim;
  }

  Gaussian_policy(const vector <Uint>& start, const ActionInfo*const aI,
    const Rvec&out) : aInfo(aI), start_mean(start[0]),
    start_prec(start.size()>1 ? start[1] : 0), nA(aI->dim), netOutputs(out),
    mean(extract_mean()), stdev(extract_stdev()),
    variance(extract_variance()), precision(extract_precision()) {}

private:
  inline Rvec extract_mean() const
  {
    assert(netOutputs.size() >= start_mean + nA);
    return Rvec(&(netOutputs[start_mean]),&(netOutputs[start_mean+nA]));
  }
  inline Rvec extract_precision() const
  {
    Rvec ret(nA);
    assert(variance.size() == nA);
    for (Uint j=0; j<nA; j++) ret[j] = 1/variance[j];
    return ret;
  }
  inline Rvec extract_variance() const
  {
    Rvec ret(nA);
    assert(stdev.size() == nA);
    for(Uint i=0; i<nA; i++) ret[i] = stdev[i]*stdev[i];
    return ret;
  }
  inline Rvec extract_stdev() const
  {
    if(start_prec == 0) return Rvec (nA, ACER_CONST_STDEV);
    Rvec ret(nA);
    assert(netOutputs.size() >= start_prec + nA);
    for(Uint i=0; i<nA; i++) ret[i] = precision_func(netOutputs[start_prec+i]);
    return ret;
  }

public:
  static inline Real precision_func(const Real val) {
    return 0.5*(val + std::sqrt(val*val+1)) + nnEPS;
  }
  static inline Real precision_func_diff(const Real val) {
    //return safeExp(val);
    return 0.5*(1.+val/std::sqrt(val*val+1));
  }
  static inline Real precision_inverse(const Real val) {
    //return std::log(val);
    return (val*val -.25)/val;
  }
  static void setInitial_noStdev(const ActionInfo* const aI, Rvec& initBias)
  {
    for(Uint e=0; e<aI->dim; e++) initBias.push_back(0);
  }
  static void setInitial_Stdev(const ActionInfo* const aI, Rvec& initBias, const Real std0)
  {
    for(Uint e=0; e<aI->dim; e++) initBias.push_back(precision_inverse(std0));
  }
  inline void prepare(const Rvec& unbact, const Rvec& beta)
  {
    sampAct = map_action(unbact);
    sampLogPonPolicy = logProbability(sampAct);
    sampLogPBehavior = evalBehavior(sampAct, beta);
    const Real logW = sampLogPonPolicy - sampLogPBehavior;
    sampImpWeight = std::exp( logW );
    sampRhoWeight = std::exp(logW * acerTrickPow);
  }

  static inline Real evalBehavior(const Rvec& act, const Rvec& beta)
  {
    Real p = 0;
    for(Uint i=0; i<act.size(); i++) {
      assert(beta[act.size()+i]>0);
      const Real stdi = beta[act.size()+i];
      p -= std::pow(act[i]-beta[i],2)/(stdi*stdi);
      p -= std::log(2*M_PI*stdi*stdi);
    }
    return 0.5*p;
  }

  static inline Rvec sample(mt19937*const gen, const Rvec& beta)
  {
    assert(beta.size() / 2 > 0 && beta.size() % 2 == 0);
    Rvec ret(beta.size()/2);
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);

    for(Uint i=0; i<beta.size()/2; i++) {
      Real samp = dist(*gen);
      if (samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) samp = safety(*gen);
      //     if (samp >  NORMDIST_MAX) samp =  2*NORMDIST_MAX -samp;
      //else if (samp < -NORMDIST_MAX) samp = -2*NORMDIST_MAX -samp;
      ret[i] = beta[i] + beta[beta.size()/2 + i]*samp;
    }
    return ret;
  }
  inline Rvec sample(mt19937*const gen) const
  {
    Rvec ret(nA);
    std::normal_distribution<Real> dist(0, 1);
    std::uniform_real_distribution<Real> safety(-NORMDIST_MAX, NORMDIST_MAX);

    for(Uint i=0; i<nA; i++) {
      Real samp = dist(*gen);
      if (samp >  NORMDIST_MAX || samp < -NORMDIST_MAX) samp = safety(*gen);
      //     if (samp >  NORMDIST_MAX) samp =  2*NORMDIST_MAX -samp;
      //else if (samp < -NORMDIST_MAX) samp = -2*NORMDIST_MAX -samp;
      ret[i] = mean[i] + stdev[i]*samp;
    }
    return ret;
  }

  inline Real logProbability(const Rvec& act) const
  {
    Real p = 0;
    for(Uint i=0; i<nA; i++) {
      p -= std::pow(act[i]-mean[i],2)*precision[i];
      p += std::log(0.5*precision[i]/M_PI);
    }
    return 0.5*p;
  }

  inline Rvec control_grad(const Quadratic_term*const adv, const Real eta) const
  {
    Rvec ret(nA*2, 0);
    for (Uint j=0; j<nA; j++) {
      for (Uint i=0; i<nA; i++)
        ret[j] += eta *adv->matrix[nA*j+i] *(adv->mean[i] - mean[i]);

      ret[j+nA] = .5*eta * adv->matrix[nA*j+j]*variance[j]*variance[j];
    }
    return ret;
  }

  inline Rvec policy_grad(const Rvec& act, const Real factor) const
  {
    /*
      this function returns factor * grad_phi log(policy(a,s))
      assumptions:
        - we deal with diagonal covariance matrices
        - network outputs the inverse of diag terms of the cov matrix
      Therefore log of distrib becomes:
      sum_i( -.5*log(2*M_PI*Sigma_i) -.5*(a-pi)^2*Sigma_i^-1 )
     */
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      ret[i]    = factor*(act[i]-mean[i])*precision[i];
      ret[i+nA] = factor*(std::pow(act[i]-mean[i],2)*precision[i]-1)/stdev[i];
    }
    return ret;
  }

  inline Rvec div_kl_grad(const Gaussian_policy*const pol_hat, const Real fac = 1) const {
    const Rvec vecTarget = pol_hat->getVector();
    return div_kl_grad(vecTarget, fac);
  }
  inline Rvec div_kl_grad(const Rvec& beta, const Real fac = 1) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      const Real preci = 1/std::pow(beta[nA+i],2);
      ret[i]   = fac * (mean[i]-beta[i])*preci;
      ret[i+nA]= fac * (preci-precision[i])*stdev[i];
    }
    return ret;
  }

  inline Real kl_divergence(const Gaussian_policy*const pol_hat) const {
    const Rvec vecTarget = pol_hat->getVector();
    return kl_divergence(vecTarget);
  }
  inline Real kl_divergence(const Rvec& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      const Real prech = 1/std::pow(beta[nA+i],2), R = variance[i]*prech;
      ret += R -1 -std::log(R) +std::pow(mean[i]-beta[i],2)*prech;
    }
    return 0.5*ret;
  }

  inline void finalize_grad(const Rvec&grad, Rvec&netGradient) const
  {
    assert(netGradient.size()>=start_mean+nA && grad.size() == 2*nA);
    for (Uint j=0; j<nA; j++) {
      netGradient[start_mean+j] = grad[j];
      //if bounded actions pass through tanh!
      //helps against NaNs in converting from bounded to unbounded action space:
      if(aInfo->bounded[j])  {
        if(mean[j]> BOUNDACT_MAX && grad[j]>0) netGradient[start_mean+j] = 0;
        else
        if(mean[j]<-BOUNDACT_MAX && grad[j]<0) netGradient[start_mean+j] = 0;
      }
    }

    for (Uint j=0, iS=start_prec; j<nA && start_prec != 0; j++, iS++) {
      assert(netGradient.size()>=start_prec+nA);
      netGradient[iS] = grad[j+nA] * precision_func_diff(netOutputs[iS]);
    }
  }
  inline Rvec finalize_grad(const Rvec&grad) const {
    Rvec ret = grad;
    for (Uint j=0; j<nA; j++) if(aInfo->bounded[j]) {
      if(mean[j]> BOUNDACT_MAX && grad[j]>0) ret[j]=0;
      else
      if(mean[j]<-BOUNDACT_MAX && grad[j]<0) ret[j]=0;
    }

    if(start_prec != 0)
    for (Uint j=0, iS=start_prec; j<nA; j++, iS++)
      ret[j+nA] = grad[j+nA] * precision_func_diff(netOutputs[iS]);
    return ret;
  }

  inline Rvec getMean() const {
    return mean;
  }
  inline Rvec getPrecision() const {
    return precision;
  }
  inline Rvec getStdev() const {
    return stdev;
  }
  inline Rvec getVariance() const {
    return variance;
  }
  inline Rvec getBest() const {
    return mean;
  }
  inline Rvec finalize(const bool bSample, mt19937*const gen, const Rvec& beta)
  { //scale back to action space size:
    sampAct = bSample ? sample(gen, beta) : mean;
    return aInfo->getScaled(sampAct);
  }

  inline Rvec getVector() const {
    Rvec ret = getMean();
    ret.insert(ret.end(), stdev.begin(), stdev.end());
    return ret;
  }

  void test(const Rvec& act, const Rvec& beta) const;
};
