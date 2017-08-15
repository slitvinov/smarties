/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Gaussian_policy.h"

struct Quadratic_advantage: public Quadratic_term
{
  const Gaussian_policy* const policy;

  //Normalized quadratic advantage around policy mean
  Quadratic_advantage(Uint _startMat, Uint _nA, Uint _nL,
      const vector<Real>& out, const Gaussian_policy*const pol) :
      Quadratic_term(_startMat,         0,_nA,_nL,out,pol->mean),policy(pol) {}

  //Normalized quadratic advantage, with own mean
  Quadratic_advantage(Uint _startMat, Uint _startMean, Uint _nA, Uint _nL,
      const vector<Real>& out, const Gaussian_policy*const pol) :
      Quadratic_term(_startMat,_startMean,_nA,_nL,out),policy(pol) {}

  //Non-normalized quadratic advantage, with own mean
  Quadratic_advantage(Uint _startMat, Uint _startMean, Uint _nA, Uint _nL,
      const vector<Real>& out) :
      Quadratic_term(_startMat,_startMean,_nA,_nL,out),policy(nullptr) {}

public:
  inline void grad(const vector<Real>&act, const Real Qer,
    vector<Real>& netGradient, const vector<bool>& bounded) const
  {
    assert(act.size()==nA);
    vector<Real> dErrdP(nA*nA), dPol(nA, 0), dAct(nA);
    for (Uint j=0; j<nA; j++) dAct[j] = act[j] - mean[j];

    if(policy not_eq nullptr)
    for (Uint j=0; j<nA; j++) dPol[j] = policy->mean[j] - mean[j];

    for (Uint j=0; j<nA; j++)
    for (Uint i=0; i<nA; i++) {
      const Real dOdPij = -.5*dAct[i]*dAct[j] + .5*dPol[i]*dPol[j]
        +.5*(i==j && policy not_eq nullptr ? policy->variance[i] : 0);

      dErrdP[nA*j+i] = Qer*dOdPij;
    }
    grad_matrix(dErrdP, netGradient);

    if(start_mean>0) {
      assert(netGradient.size() >= start_mean+nA);
      for (Uint a=0; a<nA; a++) {
        Real val = 0;
        for (Uint i=0; i<nA; i++)
          val += Qer * matrix[nA*a + i] * (dAct[i]-dPol[i]);

        netGradient[start_mean+a] = val;
        if(bounded[a])
        {
          if(mean[a]> BOUNDACT_MAX && netGradient[start_mean+a]>0)
            netGradient[start_mean+a] = 0;
          else
          if(mean[a]<-BOUNDACT_MAX && netGradient[start_mean+a]<0)
            netGradient[start_mean+a] = 0;
        }
      }
    }
  }

  inline void grad_unb(const vector<Real>&act, const Real Qer,
    vector<Real>& netGradient) const
  {
    assert(act.size()==nA);
    vector<Real> dErrdP(nA*nA), dPol(nA, 0), dAct(nA);
    for (Uint j=0; j<nA; j++) dAct[j] = act[j] - mean[j];

    if(policy not_eq nullptr)
    for (Uint j=0; j<nA; j++) dPol[j] = policy->mean[j] - mean[j];

    for (Uint j=0; j<nA; j++)
    for (Uint i=0; i<nA; i++) {
      const Real dOdPij = -.5*dAct[i]*dAct[j] + .5*dPol[i]*dPol[j]
        +.5*(i==j && policy not_eq nullptr ? policy->variance[i] : 0);

      dErrdP[nA*j+i] = Qer*dOdPij;
    }
    grad_matrix(dErrdP, netGradient);

    if(start_mean>0) {
      assert(netGradient.size() >= start_mean+nA);
      for (Uint a=0; a<nA; a++) {
        Real val = 0;
        for (Uint i=0; i<nA; i++)
          val += Qer * matrix[nA*a + i] * (dAct[i]-dPol[i]);

        netGradient[start_mean+a] = val;
      }
    }
  }

  inline Real computeAdvantage(const vector<Real>& action) const
  {
    Real ret = -quadraticTerm(action);
    if(policy not_eq nullptr)
    { //subtract expectation from advantage of action
      ret += quadraticTerm(policy->mean);
      for(Uint i=0; i<nA; i++)
        ret += matrix[nA*i+i] * policy->variance[i];
    }
    return 0.5*ret;
  }

  inline vector<Real> getMean() const
  {
    return mean;
  }
  inline vector<Real> getMatrix() const
  {
    return matrix;
  }

  inline Real advantageVariance() const
  {
    if(policy == nullptr) return 0;
    vector<Real> PvarP(nA*nA, 0);
    for (Uint j=0; j<nA; j++)
    for (Uint i=0; i<nA; i++)
    for (Uint k=0; k<nA; k++) {
      const Uint k1 = nA*j + k;
      const Uint k2 = nA*k + i;
      PvarP[nA*j+i] += matrix[k1] * policy->variance[k] * matrix[k2];
    }
    Real ret = quadMatMul(policy->mean, PvarP);
    for (Uint i=0; i<nA; i++)
      ret += 0.5 * PvarP[nA*i+i] * policy->variance[i];
    return ret;
  }

  void test(const vector<Real>& act);
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
