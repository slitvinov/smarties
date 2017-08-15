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
  const Uint start_mean, start_prec, nA;
  const vector<Real>& netOutputs;
  const vector<Real> mean, precision, variance, stdev;

  //Gaussian_policy with fixed variance
  Gaussian_policy(Uint _startM, Uint _nA, const vector<Real>& out) :
      start_mean(_startM), start_prec(0), nA(_nA), netOutputs(out),
      mean(extract_mean()), precision(extract_precision()),
      variance(extract_variance()), stdev(extract_stdev()) {}

  Gaussian_policy(Uint _startM, Uint _startP, Uint _nA,const vector<Real>&out) :
      start_mean(_startM), start_prec(_startP), nA(_nA), netOutputs(out),
      mean(extract_mean()), precision(extract_precision()),
      variance(extract_variance()), stdev(extract_stdev()) {}

private:
  inline vector<Real> extract_mean() const
  {
    assert(netOutputs.size() >= start_mean + nA);
    return vector<Real>(&(netOutputs[start_mean]),&(netOutputs[start_mean])+nA);
  }
  inline vector<Real> extract_precision() const
  {
    if(start_prec == 0) return vector<Real> (nA, ACER_CONST_PREC);
    vector<Real> ret(nA);
    assert(netOutputs.size() >= start_prec + nA);
    for (Uint j=0; j<nA; j++) {
      ret[j] = precision_func(netOutputs[start_prec+j]);
      assert(ret[j]>0);
    }
    return ret;
  }
  inline vector<Real> extract_variance() const
  {
    vector<Real> ret(nA);
    assert(precision.size() == nA);
    for(Uint i=0; i<nA; i++) ret[i] = 1./precision[i];
    return ret;
  }
  inline vector<Real> extract_stdev() const
  {
    vector<Real> ret(nA);
    assert(variance.size() == nA);
    for(Uint i=0; i<nA; i++) ret[i] = std::sqrt(variance[i]);
    return ret;
  }
  static inline Real precision_func(const Real val)
  {
    return safeExp(val);
    //return 0.5*(val + std::sqrt(val*val+1));
  }
  static inline Real precision_func_diff(const Real val)
  {
    return safeExp(val);
    //return 0.5*(1.+val/std::sqrt(val*val+1));
  }

public:
  static inline Real evalBehavior(const vector<Real>& act, const vector<Real>& beta)
  {
    Real p = 0;
    for(Uint i=0; i<act.size(); i++) {
      assert(beta[act.size()+i]>0);
      const Real stdi = beta[act.size()+i];
      p -= 0.5*(act[i]-beta[i])*(act[i]-beta[i])/(stdi*stdi);
      p -= 0.5*std::log(2*M_PI*stdi*stdi);
    }
    return p;
  }

  inline Real evalLogProbability(const vector<Real>& act) const
  {
    Real p = 0;
    for(Uint i=0; i<nA; i++) {
      p -= 0.5*precision[i]*(act[i]-mean[i])*(act[i]-mean[i]);
      p += 0.5*std::log(0.5*precision[i]/M_PI);
    }
    return p;
  }

  inline vector<Real> sample(mt19937*const gen) const
  {
    std::vector<Real> ret(nA);
    for(Uint i=0; i<nA; i++) {
      std::normal_distribution<Real> dist(mean[i], stdev[i]);
      ret[i] = dist(*gen);
    }
    return ret;
  }

  inline vector<Real> control_grad(const Quadratic_term*const adv, const Real eta) const
  {
    vector<Real> ret(nA*2, 0);
    for (Uint j=0; j<nA; j++)
    {
      for (Uint i=0; i<nA; i++)
        ret[j] += eta *adv->matrix[nA*j+i] *(adv->mean[i] - mean[i]);

      ret[j+nA] = .5*eta * adv->matrix[nA*j+j]*variance[j]*variance[j];
    }
    return ret;
  }

  inline vector<Real> policy_grad(const vector<Real>& act, const Real factor) const
  {
    /*
      this function returns factor * grad_phi log(policy(a,s))
      assumptions:
        - we deal with diagonal covariance matrices
        - network outputs the inverse of diag terms of the cov matrix
      Therefore log of distrib becomes:
      sum_i( -.5*log(2*M_PI*Sigma_i) -.5*(a-pi)^2*Sigma_i^-1 )
     */
    vector<Real> ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      ret[i]    = factor*(act[i]-mean[i])*precision[i];
      ret[i+nA] =.5*factor*(variance[i] -std::pow(act[i]-mean[i],2));
    }
    return ret;
  }

  inline vector<Real> div_kl_grad(const Gaussian_policy*const pol_hat, const Real fac = 1) const
  {
    /*
      Div_KL between two multiv. Gaussians N_1 and N_2 of dim=M is
      0.5*( trace(inv(Sigma_2)*Sigma_1) + (m_2 - m_1)'*inv(Sigma_2)*(m_2 - m_1) - M + ln(det(Sigma_2)/det(Sigma_1))
      assumptions:
        - we deal with diagonal covariance matrices
        - network outputs the inverse of diagonal terms of the cov matrix

      therefore divKL assumes shape
      0.5*(\sum_i( Sigma_1_i*(Sigma_2_i)^-1 + (m_2_i - m_1_i)^2*(Sigma_2_i)^-1 -1 +ln(Sigma_2_i) -ln(Sigma_1_i))
     */
    vector<Real> ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      ret[i]    = fac*(mean[i]-pol_hat->mean[i])*precision[i];

      //               v from trace        v from quadratic term
      ret[i+nA] = .5*fac*(pol_hat->variance[i] - variance[i]
              + std::pow(mean[i] - pol_hat->mean[i], 2) );
      //          ^ from normalization
    }
    return ret;
  }

  inline void finalize_grad(const vector<Real>&grad, vector<Real>&netGradient, const vector<bool>& bounded) const
  {
    assert(netGradient.size()>=start_mean+nA && grad.size() == 2*nA);
    for (Uint j=0; j<nA; j++) {
      //#ifdef ACER_BOUNDED //clip derivative
      //if(bounded[j]) {
      //  const Real MAX = ACER_MAX_ACT-mean[j], MIN = -ACER_MAX_ACT-mean[j];
      //  netGradient[start_mean+j] = clip(grad[j], MAX, MIN);
      //} else
      //#endif
        netGradient[start_mean+j] = grad[j];
    }

    if(start_prec != 0)
    for (Uint j=0; j<nA; j++) {
      const Uint iS = start_prec+j;
      assert(netGradient.size()>=start_prec+nA);
      const Real diff = precision_func_diff(netOutputs[iS]);
      //#ifdef ACER_BOUNDED //clip derivative
      //if(bounded[j]) {
      //  const Real M=ACER_MAX_PREC-precision[j], m=ACER_MIN_PREC-precision[j];
      //  netGradient[iS] = clip(grad[j+nA], M, m) * diff;
      //} else //minimum precision is unbounded
        netGradient[iS] = min(grad[j+nA], ACER_MAX_PREC-precision[j]) * diff;
      //#else
      //netGradient[iS] = grad[j+nA] * diff;
      //#endif
    }
  }

  inline void finalize_grad_unb(const vector<Real>&grad, vector<Real>&netGradient) const
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

  inline Real kl_divergence(const Gaussian_policy*const pol_hat) const
  {
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      ret += precision[i]/pol_hat->precision[i] - 1;
      ret += std::pow(mean[i]-pol_hat->mean[i],2)*precision[i];
      ret += std::log(pol_hat->precision[i])-std::log(precision[i]);
    }
    return 0.5*ret;
  }

  inline vector<Real> getMean() const
  {
    return mean;
  }
  inline vector<Real> getPrecision() const
  {
    return precision;
  }
  inline vector<Real> getStdev() const
  {
    return stdev;
  }
  inline vector<Real> getVariance() const
  {
    return variance;
  }

  void test(const vector<Real>& act, const Gaussian_policy*const pol_hat,
      const Quadratic_term*const a);
};
