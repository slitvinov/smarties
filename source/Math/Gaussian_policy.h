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
  const Rvec netOutputs;
  const Rvec mean, precision, variance, stdev;

  Rvec sampAct;
  Real sampLogPonPolicy=0, sampLogPBehavior=0, sampImpWeight=0, sampRhoWeight=0;

  Gaussian_policy(const vector <Uint>& start, const ActionInfo*const aI,
    const Rvec&out) : aInfo(aI), start_mean(start[0]),
    start_prec(start.size()>1 ? start[1] : 0), nA(aI->dim), netOutputs(out),
    mean(extract_mean()), precision(extract_precision()),
    variance(extract_variance()), stdev(extract_stdev()) {}

private:
  inline Rvec extract_mean() const
  {
    assert(netOutputs.size() >= start_mean + nA);
    return Rvec(&(netOutputs[start_mean]),&(netOutputs[start_mean])+nA);
  }
  inline Rvec extract_precision() const
  {
    if(start_prec == 0) return Rvec (nA, ACER_CONST_PREC);
    Rvec ret(nA);
    assert(netOutputs.size() >= start_prec + nA);
    for (Uint j=0; j<nA; j++) {
      ret[j] = precision_func(netOutputs[start_prec+j]);
      assert(ret[j]>0);
    }
    return ret;
  }
  inline Rvec extract_variance() const
  {
    Rvec ret(nA);
    assert(precision.size() == nA);
    for(Uint i=0; i<nA; i++) ret[i] = 1./precision[i];
    return ret;
  }
  inline Rvec extract_stdev() const
  {
    Rvec ret(nA);
    assert(variance.size() == nA);
    for(Uint i=0; i<nA; i++) ret[i] = std::sqrt(variance[i]);
    return ret;
  }
  static inline Real precision_func(const Real val)
  {
    //return safeExp(val) + numeric_limits<Real>::epsilon();
    return 0.5*(val + std::sqrt(val*val+1));
  }
  static inline Real precision_func_diff(const Real val)
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
    sampImpWeight = safeExp( logW );
    sampRhoWeight = std::exp(logW / std::sqrt(nA));
  }

  static inline Real evalBehavior(const Rvec& act, const Rvec& beta)
  {
    Real p = 0;
    for(Uint i=0; i<act.size(); i++) {
      assert(beta[act.size()+i]>0);
      const Real stdi = beta[act.size()+i];
      p -= (act[i]-beta[i])*(act[i]-beta[i])/(stdi*stdi);
      p -= std::log(2*M_PI*stdi*stdi);
    }
    return 0.5*p;
  }

  static inline Rvec sample(mt19937*const gen, const Rvec& beta)
  {
    assert(beta.size() / 2 > 0 && beta.size() % 2 == 0);
    Rvec ret(beta.size()/2);
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<beta.size()/2; i++) {
      Real samp = dist(*gen);
           if (samp >  NORMDIST_MAX) samp =  2*NORMDIST_MAX -samp;
      else if (samp < -NORMDIST_MAX) samp = -2*NORMDIST_MAX -samp;
      ret[i] = beta[i] + beta[beta.size()/2 + i]*samp;
    }
    return ret;
  }
  inline Rvec sample(mt19937*const gen) const
  {
    Rvec ret(nA);
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<nA; i++) {
      Real samp = dist(*gen);
           if (samp >  NORMDIST_MAX) samp =  2*NORMDIST_MAX -samp;
      else if (samp < -NORMDIST_MAX) samp = -2*NORMDIST_MAX -samp;
      ret[i] = mean[i] + stdev[i]*samp;
    }
    return ret;
  }

  inline Real evalLogProbability(const Rvec& act) const
  {
    Real p = 0;
    for(Uint i=0; i<nA; i++) {
      p -= 0.5*precision[i]*(act[i]-mean[i])*(act[i]-mean[i]);
      p += 0.5*std::log(0.5*precision[i]/M_PI);
    }
    return p;
  }

  inline Rvec control_grad(const Quadratic_term*const adv, const Real eta) const
  {
    Rvec ret(nA*2, 0);
    for (Uint j=0; j<nA; j++)
    {
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
      ret[i+nA] =.5*factor*(variance[i] -std::pow(act[i]-mean[i],2));
    }
    return ret;
  }

  inline Rvec policy_grad(const Rvec& act, const Rvec& beta, const Real factor) const
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
      const Real vari = beta[nA + i]*beta[nA + i];
      ret[i]    = factor*(act[i]-beta[i])/vari;
      ret[i+nA] =.5*factor*(vari -std::pow(act[i]-beta[i],2));
    }
    return ret;
  }

  inline Rvec div_kl_grad(const Gaussian_policy*const pol_hat, const Real fac = 1) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      ret[i]    = fac*(mean[i]-pol_hat->mean[i])*precision[i];

      ret[i+nA] = .5*fac*(pol_hat->variance[i] - variance[i]
              + std::pow(mean[i] - pol_hat->mean[i], 2) );
    }
    return ret;
  }
  inline Rvec div_kl_opp_grad(const Gaussian_policy*const pol_hat, const Real fac = 1) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      ret[i]   =fac*(mean[i]-pol_hat->mean[i])*pol_hat->precision[i];
      ret[i+nA]=.5*fac*(precision[i]-pol_hat->precision[i])*pow(variance[i],2);
    }
    return ret;
  }
  inline Rvec div_kl_opp_grad(const Rvec& beta, const Real fac = 1) const
  {
    Rvec ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      const Real preci = 1/std::pow(beta[nA+i],2);
      ret[i]   = fac  * (mean[i]-beta[i])*preci;
      ret[i+nA]= fac/2* (precision[i]-preci)*std::pow(variance[i],2);
    }
    return ret;
  }
  inline Real kl_divergence(const Gaussian_policy*const pol_hat) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      ret += precision[i]*pol_hat->variance[i] - 1;
      ret += std::pow(mean[i]-pol_hat->mean[i],2)*precision[i];
      ret += std::log(pol_hat->precision[i]*variance[i]);
    }
    return 0.5*ret;
  }
  inline Real kl_divergence_opp(const Gaussian_policy*const pol_hat) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      ret += pol_hat->precision[i]*variance[i] - 1;
      ret += std::pow(mean[i]-pol_hat->mean[i],2)*pol_hat->precision[i];
      ret += std::log(precision[i]*pol_hat->variance[i]);
    }
    return 0.5*ret;
  }
  inline Real kl_divergence_opp(const Rvec& beta) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      ret += variance[i]/std::pow(beta[nA+i],2) - 1;
      ret += std::pow(mean[i]-beta[i],2)/std::pow(beta[nA+i],2);
      ret += std::log(precision[i]*std::pow(beta[nA+i],2));
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
      if(aInfo->bounded[j])
      {
        if(mean[j]> BOUNDACT_MAX && netGradient[start_mean+j]>0)
          netGradient[start_mean+j] = 0;
        else
        if(mean[j]<-BOUNDACT_MAX && netGradient[start_mean+j]<0)
          netGradient[start_mean+j] = 0;
      }
    }

    if(start_prec != 0)
    for (Uint j=0; j<nA; j++) {
      const Uint iS = start_prec+j;
      assert(netGradient.size()>=start_prec+nA);
      const Real diff = precision_func_diff(netOutputs[iS]);
      if(precision[j]>ACER_MAX_PREC && grad[j+nA]>0) netGradient[iS] = 0;
      else netGradient[iS] = grad[j+nA] * diff;
      //#ifdef ACER_BOUNDED //clip derivative
      //if(bounded[j]) {
      //  const Real M=ACER_MAX_PREC-precision[j], m=ACER_MIN_PREC-precision[j];
      //  netGradient[iS] = clip(grad[j+nA], M, m) * diff;
      //} else //minimum precision is unbounded
      //  netGradient[iS] = min(grad[j+nA], ACER_MAX_PREC-precision[j]) * diff;
      //#endif
    }
  }

  inline Rvec finalize_grad(const Rvec&grad) const
  {
    Rvec ret = grad;
    for (Uint j=0; j<nA; j++)
    if(aInfo->bounded[j]) {
      if(mean[j]> BOUNDACT_MAX && grad[j]>0) ret[j]=0;
      else
      if(mean[j]<-BOUNDACT_MAX && grad[j]<0) ret[j]=0;
    }

    if(start_prec != 0)
    for (Uint j=0; j<nA; j++) {
      const Uint iS = start_prec+j;
      const Real diff = precision_func_diff(netOutputs[iS]);
      if(precision[j]>ACER_MAX_PREC && grad[j+nA]>0) ret[j+nA] = 0;
      else ret[j+nA] = grad[j+nA] * diff;
    }
    return ret;
  }

  inline void finalize_grad_unb(const Rvec&grad, Rvec&netGradient) const
  {
    assert(netGradient.size()>=start_mean+nA && grad.size() == 2*nA);
    for (Uint j=0; j<nA; j++)  netGradient[start_mean+j] = grad[j];

    if(start_prec != 0)
    for (Uint j=0; j<nA; j++) {
      assert(netGradient.size()>=start_prec+nA);
      const Real diff = precision_func_diff(netOutputs[start_prec+j]);
      netGradient[start_prec+j] = grad[j+nA] * diff;
    }
  }
  inline Rvec getMean() const
  {
    return mean;
  }
  inline Rvec getPrecision() const
  {
    return precision;
  }
  inline Rvec getStdev() const
  {
    return stdev;
  }
  inline Rvec getVariance() const
  {
    return variance;
  }
  inline Rvec getBest() const
  {
    return mean;
  }
  inline Rvec finalize(const bool bSample, mt19937*const gen, const Rvec& beta)
  { //scale back to action space size:
    sampAct = bSample ? sample(gen, beta) : mean;
    return aInfo->getScaled(sampAct);
  }

  inline Rvec getBeta() const
  {
    Rvec ret = getStdev();
    ret.insert(ret.begin(), mean.begin(), mean.end());
    return ret;
  }
  static inline void anneal_beta(Rvec& beta, const Real eps)
  {
    assert(beta.size() / 2 > 0 && beta.size() % 2 == 0);
    const Real safety_std = std::sqrt(1/ACER_MAX_PREC);
    for(Uint i=beta.size()/2; i<beta.size(); i++)
      beta[i] = std::max(safety_std + eps, beta[i]);
  }

  inline Rvec map_action(const Rvec& sent) const
  {
    return aInfo->getInvScaled(sent);
  }
  static inline Uint compute_nA(const ActionInfo* const aI)
  {
    assert(aI->dim);
    return aI->dim;
  }
  void test(const Rvec& act, const Gaussian_policy*const pol_hat) const;//,
      //const Quadratic_term*const a);
};
