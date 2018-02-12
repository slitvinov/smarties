/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Gaussian_mixture.h"

template<Uint nExperts>
struct Mixture_advantage
{
  template <typename T>
  static inline string print(const array<T,nExperts> vals) {
    std::ostringstream o;
    for (Uint i=0; i<nExperts-1; i++) o << vals[i] << " ";
    o << vals[nExperts-1];
    return o.str();
  }

  Rvec getParam() const {
    Rvec ret = matrix[0];
    for(Uint e=1; e<nExperts; e++)
      ret.insert(ret.end(), matrix[e].begin(), matrix[e].end());
    for(Uint e=0; e<nExperts; e++) ret.push_back(coef[e]);
    return ret;
  }

  static void setInitial(const ActionInfo* const aI, Rvec& initBias) {
    for(Uint e=0; e<nExperts; e++) initBias.push_back(-1);
    for(Uint e=nExperts; e<compute_nL(aI); e++) initBias.push_back(1);
  }

  const Uint start_matrix, start_coefs, nA, nL;
  const Rvec netOutputs;
  const array<Real, nExperts> coef;
  const array<Rvec, nExperts> matrix;
  const ActionInfo* const aInfo;
  const Gaussian_mixture<nExperts>* const policy;

  //Normalized quadratic advantage, with own mean
  Mixture_advantage(const vector<Uint>& starts, const ActionInfo* const aI,
   const Rvec& out, const Gaussian_mixture<nExperts>*const pol) :
   start_matrix(starts[0]+nExperts), start_coefs(starts[0]), nA(aI->dim),
   nL(compute_nL(aI)), netOutputs(out), coef(extract_coefs()),
   matrix(extract_matrix()), aInfo(aI), policy(pol) {}

private:
  inline array<Rvec, nExperts> extract_matrix() const {
    array<Rvec, nExperts> ret;
    for (Uint e=0; e<nExperts; e++) {
      ret[e] = Rvec(2*nA);
      for(Uint i=0; i<2*nA; i++)
        ret[e][i] = diag_func(netOutputs[start_matrix +2*nA*e +i]);
    }
    return ret;
  }
  inline array<Real,nExperts> extract_coefs() const  {
    array<Real, nExperts> ret;
    for(Uint e=0; e<nExperts;e++) ret[e]=diag_func(netOutputs[start_coefs+e]);
    return ret;
  }

  inline void grad_matrix(Rvec& G, const Real err) const {
    for (Uint e=0; e<nExperts; e++) {
      G[start_coefs+e] *= err * diag_func_diff(netOutputs[start_coefs+e]);
      for (Uint i=0, ind=start_matrix +2*nA*e; i<2*nA; i++, ind++)
         G[ind] *= err * diag_func_diff(netOutputs[ind]);
    }
  }

  static inline Real diag_func(const Real val) {
    return 0.5*(val + std::sqrt(val*val+1));
    //return safeExp(val);
  }
  static inline Real diag_func_diff(const Real val) {
    return 0.5*(1 + val/std::sqrt(val*val+1));
    //return safeExp(val);
  }
  static inline Real offdiag_func(const Real val) { return val; }
  static inline Real offdiag_func_diff(Real val) { return 1; }

public:

  inline Real computeAdvantage(const Rvec& act) const {
    Real ret = 0;
    for (Uint e=0; e<nExperts; e++) {
      const Real shape = -.5 * diagInvMul(act, matrix[e], policy->means[e]);
      ret += policy->experts[e] * coef[e] * std::exp(shape);
      #if 1
        const array<Real,nExperts> expectations = expectation(e);
        for (Uint E=0; E<nExperts; E++) ret += coef[e] * expectations[E];
      #endif
    }
    return ret;
  }

  inline Real coefMixRatio(const Rvec&A, const Rvec&VAR) const {
    Real ret1 = 1, ret2 = 1;
    for (Uint i=0; i<nA; i++) {
      ret1 *= A[i]   /(A[i]   +VAR[i]);
      ret2 *= A[i+nA]/(A[i+nA]+VAR[i]);
    }
    return .5*(std::sqrt(ret1)+std::sqrt(ret2));
  }

  inline array<Real,nExperts> expectation(const Uint expert) const {
    array<Real, nExperts> ret;
    for (Uint E=0; E<nExperts; E++) {
      const Real W = policy->experts[expert] * policy->experts[E];
      const Real ratio = coefMixRatio(matrix[expert], policy->variances[E]);
      if(E==expert) ret[E] = - W * ratio;
      else {
        const Real overl = diagInvMulVar(policy->means[E],
        matrix[expert], policy->means[expert], policy->variances[E]);
        ret[E] = - W * ratio * std::exp(-0.5*overl);
      }
    }
    return ret;
  }

  inline void grad(const Rvec&a, const Real Qer, Rvec& G) const
  {
    assert(a.size()==nA);
    for (Uint e=0; e<nExperts; e++)
    {
      const Real shape = -.5 * diagInvMul(a, matrix[e], policy->means[e]);
      const Real orig = policy->experts[e] * std::exp(shape);
      G[start_coefs+e] = orig;

      #if 1
      array<Real, nExperts> fact1, fact2, overl, W;
      for (Uint E=0; E<nExperts; E++)
      {
        fact1[E] = 1; fact2[E] = 1;
        overl[E] = e==E? 1 : std::exp( -0.5*diagInvMulVar(policy->means[E],
          matrix[e], policy->means[e], policy->variances[E]) );
        W[E] = - policy->experts[e] * policy->experts[E];
        for (Uint i=0; i<nA; i++) {
          fact1[E] *= matrix[e][i]   /(matrix[e][i]   +policy->variances[E][i]);
          fact2[E] *= matrix[e][i+nA]/(matrix[e][i+nA]+policy->variances[E][i]);
        }
        fact1[E] = std::sqrt(fact1[E])/2; fact2[E] = std::sqrt(fact2[E])/2;
        G[start_coefs+e] += W[E] * (fact1[E]+fact2[E]) * overl[E];
      }
      #endif

      for (Uint i=0, ind=start_matrix+ 2*nA*e; i<nA; i++, ind++)
      {
        const Real m=policy->means[e][i], p1=matrix[e][i], p2=matrix[e][i+nA];
        G[ind]   = a[i]>m ? orig*coef[e] * std::pow((a[i]-m)/p1, 2)/2 : 0;
        G[ind+nA]= a[i]<m ? orig*coef[e] * std::pow((a[i]-m)/p2, 2)/2 : 0;
        #if 1
          for(Uint E=0; E<nExperts; E++)
          {
            const Real S = policy->variances[E][i], M = policy->means[E][i];
            const Real diff1 = S/(p1*(p1+S)), diff2 = S/(p2*(p2+S));
            const Real expectE = W[E]*coef[e]*(fact1[E]+fact2[E])*overl[E];
            if(M>m) G[ind] += expectE * std::pow((m-M)/(p1+S), 2)/2;
            else G[ind+nA] += expectE * std::pow((m-M)/(p2+S), 2)/2;
            G[ind]    += W[E]*coef[e]*overl[E]*fact1[E]*diff1/2;
            G[ind+nA] += W[E]*coef[e]*overl[E]*fact2[E]*diff2/2;
          }
        #endif
      }
    }
    grad_matrix(G, Qer);
  }

  static inline Uint compute_nL(const ActionInfo* const aI) {
    return nExperts*(1 + 2*aI->dim);
  }

  void test(const Rvec& act, mt19937*const gen) const
  {
    const Uint numNetOutputs = netOutputs.size();
    Rvec _grad(numNetOutputs, 0);
    grad(act, 1, _grad);
    ofstream fout("mathtest.log", ios::app);
    for(Uint i = 0; i<nL; i++)
    {
      Rvec out_1 = netOutputs, out_2 = netOutputs;
      const Uint index = start_coefs+i;
      out_1[index] -= 0.0001; out_2[index] += 0.0001;

      Mixture_advantage a1(vector<Uint>{start_coefs}, aInfo, out_1, policy);
      Mixture_advantage a2(vector<Uint>{start_coefs}, aInfo, out_2, policy);
      const Real A_1 = a1.computeAdvantage(act), A_2 = a2.computeAdvantage(act);
      const Real fdiff =(A_2-A_1)/.0002, abserr = std::fabs(_grad[index]-fdiff);
      const Real scale = std::max(std::fabs(fdiff), std::fabs(_grad[index]));
      //if(abserr>1e-7 && abserr/scale>1e-4)
      {
        cout<<"Adv grad "<<i<<" finite differences "<<fdiff<<" analytic "
          <<_grad[index]<<" error "<<abserr<<" "<<abserr/scale<<endl;
      }
    }
    fout.close();
    #if 0
    Real expect = 0, expect2 = 0;
    for(Uint i = 0; i<1e7; i++) {
      const auto sample = policy->sample(gen);
      const auto advant = computeAdvantage(sample);
      expect += advant; expect2 += std::fabs(advant);
    }
    const Real stdef = expect2/1e6, meanv = expect/1e6;
    printf("Ratio of expectation: %f, mean %f\n", meanv/stdef, meanv);
    #endif
  }

  inline Real diagInvMul(const Rvec& act,
    const Rvec& mat, const Rvec& mean) const {
    assert(act.size()==nA); assert(mean.size()==nA); assert(mat.size()==2*nA);
    Real ret = 0;
    for (Uint i=0; i<nA; i++) {
      const Uint matind = act[i]>mean[i] ? i : i+nA;
      ret += std::pow(act[i]-mean[i],2)/mat[matind];
    }
    return ret;
  }

  inline Real diagInvMulVar(const Rvec&pol, const Rvec&mat,
    const Rvec& mean, const Rvec& var) const {
    assert(pol.size()==nA); assert(mean.size()==nA); assert(mat.size()==2*nA);
    Real ret = 0;
    for(Uint i=0; i<nA; i++) {
      const Uint matind = pol[i]>mean[i] ? i : i+nA;
      ret += std::pow(pol[i]-mean[i],2)/(mat[matind]+var[i]);
    }
    return ret;
  }
};
