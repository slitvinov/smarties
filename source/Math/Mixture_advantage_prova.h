/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#if 0
#include "Gaussian_mixture_trunc.h"
#else
#include "Gaussian_mixture.h"
#endif

template<Uint nExperts>
struct Mixture_advantage
{
  template <typename T>
  static inline string print(const array<T,nExperts> vals)
  {
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

  static void setInitial(const ActionInfo* const aI, Rvec& initBias)
  {
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
  inline array<Rvec, nExperts> extract_matrix() const
  {
    array<Rvec, nExperts> ret;
    for (Uint e=0; e<nExperts; e++) {
      ret[e] = Rvec(nA);
      for(Uint i=0; i<nA; i++)
        ret[e][i] = unbPosMap_func(netOutputs[start_matrix +nA*e +i]);
    }
    return ret;
  }
  inline array<Real,nExperts> extract_coefs() const
  {
    array<Real, nExperts> ret;
    for(Uint e=0; e<nExperts;e++) ret[e]=unbPosMap_func(netOutputs[start_coefs+e]);
    return ret;
  }

  inline void grad_matrix(Rvec& netGradient, const Real err) const
  {
    for (Uint e=0; e<nExperts; e++) {
      netGradient[start_coefs+e]*=err*unbPosMap_diff(netOutputs[start_coefs+e]);
      for (Uint i=0, ind=start_matrix+nA*e; i<nA; i++, ind++)
         netGradient[ind] *= err*unbPosMap_diff(netOutputs[ind]);
    }
  }

public:
  inline array<Real,nExperts> expectation(const Uint expert) const
  {
    array<Real,nExperts> ret;
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

  inline Real computeAdvantage(const Rvec& act) const
  {
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

  inline void grad(const Rvec&a, const Real Qer, Rvec& netGradient) const
  {
    assert(a.size()==nA);
    for (Uint e=0; e<nExperts; e++) {
      const Real shape = -.5 * diagInvMul(a, matrix[e], policy->means[e]);
      const Real orig = policy->experts[e] * std::exp(shape);
      const array<Real, nExperts> expect = expectation(e);
      //printf("val = %g expect: %s\n", orig, print(expect).c_str()); fflush(0);
      netGradient[start_coefs+e] = orig;
      for(Uint E=0;E<nExperts;E++) netGradient[start_coefs+e] += expect[E];

      for (Uint i=0, ind=start_matrix+nA*e; i<nA; i++, ind++) {
        const Real p = matrix[e][i], m = policy->means[e][i];
        netGradient[ind] = orig*coef[e] * std::pow((a[i]-m)/p,2)/2;
        #if 1
        for(Uint E=0; E<nExperts; E++) {
          const Real S = policy->variances[E][i], M = policy->means[E][i];
          const Real term = S/(p*(p+S)) + std::pow((m-M)/(p+S), 2);
          netGradient[ind] += expect[E] * coef[e] * term/2;
        }
        #endif
      }
    }
    grad_matrix(netGradient, Qer);
  }

  static inline Uint compute_nL(const ActionInfo* const aI)
  {
    return nExperts*(1 + aI->dim);
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
        fout<<"Adv grad "<<i<<" finite differences "<<fdiff<<" analytic "
          <<_grad[index]<<" error "<<abserr<<" "<<abserr/scale<<endl;
      }
    }
    fout.close();
    #if 0
    Real expect = 0, expect2 = 0;
    for(Uint i = 0; i<1e6; i++) {
      const auto sample = policy->sample(gen);
      const auto advant = computeAdvantage(sample);
      expect += advant; expect2 += std::fabs(advant);
    }
    const Real stdef = expect2/1e6, meanv = expect/1e6;
    printf("Ratio of expectation: %f, mean %f\n", meanv/stdef, meanv);
    #endif
  }

  inline Real diagInvMul(const Rvec& act,
    const Rvec& mat, const Rvec& mean) const
  {
    assert(act.size()==nA);assert(mean.size()==nA);assert(mat.size()==nA);
    Real ret = 0;
    for (Uint i=0; i<nA; i++) ret += std::pow(act[i]-mean[i],2)/mat[i];
    return ret;
  }
  inline Real diagInvMulVar(const Rvec&act, const Rvec&mat,
    const Rvec& mean, const Rvec& var) const
  {
    assert(act.size()==nA);assert(mean.size()==nA);assert(mat.size()==nA);
    Real ret = 0;
    for(Uint i=0; i<nA; i++) ret += std::pow(act[i]-mean[i],2)/(mat[i]+var[i]);
    return ret;
  }
  inline Real coefMixRatio(const Rvec&A, const Rvec&S) const
  {
    Real ret = 1;
    for (Uint i=0; i<nA; i++) ret *= A[i]/(A[i]+S[i]);
    return std::sqrt(ret);
  }
};
