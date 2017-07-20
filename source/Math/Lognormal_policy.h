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

struct Lognormal_policy
{
  const Uint start_mean, start_std, nA;
  const vector<Real>& netOutputs;
  const vector<Real> mean, stdev;

  Lognormal_policy(Uint startM, Uint startS, Uint _nA, const vector<Real>&out) : start_mean(startM), start_std(startS), nA(_nA), netOutputs(out), mean(extract_mean()), stdev(extract_stdev()) {}

private:
  inline vector<Real> extract_mean() const
  {
    assert(netOutputs.size() >= start_mean + nA);
    return vector<Real>(&(netOutputs[start_mean]),&(netOutputs[start_mean])+nA);
  }

  inline vector<Real> extract_stdev() const
  {
#ifdef INTEGRATEANDFIRESHARED
    return vector<Real>(nA, std_func(netOutputs[start_std]));
#else
    vector<Real> ret(nA);
    assert(netOutputs.size() >= start_std + nA);
    for (Uint j=0; j<nA; j++) {
      ret[j] = std_func(netOutputs[start_std+j]);
      assert(ret[j]>0);
    }
    return ret;
#endif
  }
  static inline Real std_func(const Real val)
  {
    return safeExp(val) + 0.01;
    //return 0.5*(val + std::sqrt(val*val+1)) + 0.01;
  }
  static inline Real std_func_diff(const Real val)
  {
    return safeExp(val);
    //return 0.5*(1.+val/std::sqrt(val*val+1));
  }

public:
  inline vector<Real> sample(mt19937*const gen) const
  {
    std::vector<Real> ret(nA);
    for(Uint i=0; i<nA; i++) {
      std::lognormal_distribution<Real> dist(mean[i], stdev[i]);
      ret[i] = dist(*gen);
    }
    return ret;
  }

  inline vector<Real> policy_grad(const vector<Real>& act, const Real fac) const
  {
    vector<Real> ret(2*nA);
    for (Uint i=0; i<nA; i++) {
      const Real prec = 1/(stdev[i]*stdev[i]), logA = std::log(act[i]);
      ret[i]    = fac*(logA-mean[i])*prec;
      ret[i+nA] = fac*(std::pow(logA-mean[i],2)*prec/stdev[i] -1/stdev[i]);
    }
    return ret;
  }

  inline void finalize_grad(const vector<Real>&grad, vector<Real>&netGrad) const
  {
    assert(netGrad.size()>=start_mean+nA && grad.size() == 2*nA);
    for (Uint j=0; j<nA; j++) netGrad[start_mean+j] = grad[j];
#ifdef INTEGRATEANDFIRESHARED
    netGrad[start_std] = 0;
    const Real diff = std_func_diff(netOutputs[start_std]);
    for (Uint j=0; j<nA; j++) netGrad[start_std] += grad[j+nA] * diff;
#else
    for (Uint j=0; j<nA; j++)
      netGrad[start_std+j] = grad[j+nA]*std_func_diff(netOutputs[start_std+j]);
#endif
  }

  inline vector<Real> getMean() const
  {
    return mean;
  }
  inline vector<Real> getStdev() const
  {
    return stdev;
  }
};
