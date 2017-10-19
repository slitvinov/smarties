/*
 *  Settings.h
 *  rl
 *
 *  Created by Guido Novati on 02.05.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "../Bund.h"
#include "../StateAction.h"

inline vector<Real> sum3Grads(const vector<Real>& f, const vector<Real>& g,
  const vector<Real>& h)
{
  assert(g.size() == f.size());
  assert(h.size() == f.size());
  vector<Real> ret(f.size());
  for(Uint i=0; i<f.size(); i++) ret[i] = f[i]+g[i]+h[i];
  return ret;
}

inline vector<Real> sum2Grads(const vector<Real>& f, const vector<Real>& g)
{
  assert(g.size() == f.size());
  vector<Real> ret(f.size());
  for(Uint i=0; i<f.size(); i++) ret[i] = f[i]+g[i];
  return ret;
}

inline vector<Real> trust_region_split(const vector<Real>& grad, const vector<Real>& onpol, const vector<Real>& trust, const Real delta)
{
  assert(grad.size() == trust.size());
  const Uint nA = grad.size();
  vector<Real> ret(nA);
  const Real EPS = numeric_limits<Real>::epsilon();
  Real dot_KG=0, dot_KP=0, dot_PG=0, norm_K=EPS, norm_P=EPS;
  for (Uint j=0; j<nA; j++) {
    dot_KP += trust[j] *  grad[j]; dot_KG += trust[j] * onpol[j];
    dot_PG +=  grad[j] * onpol[j]; norm_K += trust[j] * trust[j];
    norm_P += onpol[j] * onpol[j];
  }
  const Real nK = std::sqrt(norm_K), nP = std::sqrt(norm_P);
  const Real dirDelta = delta/nK * std::max((Real)0, dot_KP/(nK*nP));
  const Real proj_para = std::min((Real)0, dirDelta - dot_KP/norm_K);
  const Real proj_orth = std::min((Real)0, dot_PG/norm_P);
  //#ifndef NDEBUG
  //if(proj>0) {printf("Hit DKL constraint\n");fflush(0);}
  //else {printf("Not Hit DKL constraint\n");fflush(0);}
  //#endif
  for (Uint j=0; j<nA; j++) {
    ret[j] = grad[j] + proj_para*trust[j] - proj_orth*onpol[j];
    if(ret[j]*grad[j] < 0) ret[j] = 0;
  }
  return ret;
}

inline vector<Real> trust_region_update(const vector<Real>& grad,
  const vector<Real>& trust, const Real delta)
{
  assert(grad.size() == trust.size());
  const Uint nA = grad.size();
  vector<Real> ret(nA);
  Real dot=0, norm = numeric_limits<Real>::epsilon();
  for (Uint j=0; j<nA; j++) {
    norm += trust[j] * trust[j];
    dot +=  trust[j] *  grad[j];
  }
  const Real proj = std::max((Real)0, dot/norm - delta/std::sqrt(norm));
  //#ifndef NDEBUG
  //if(proj>0) {printf("Hit DKL constraint\n");fflush(0);}
  //else {printf("Not Hit DKL constraint\n");fflush(0);}
  //#endif
  for (Uint j=0; j<nA; j++) {
    ret[j] = grad[j]-proj*trust[j];
    if(ret[j]*grad[j] < 0) ret[j] = 0;
  }
  return ret;
}

inline vector<Real> trust_region_separate(const vector<Real>& grad,
  const vector<Real>& trust, const Real delta)
{
  assert(grad.size() == trust.size());
  const Uint nA = grad.size();
  vector<Real> ret(nA);
  for (Uint j=0; j<nA; j++) {
    const Real norm = trust[j] * trust[j];
    const Real dot  = trust[j] *  grad[j];
    const Real proj = std::max((Real)0., (dot - norm*delta)/norm);
    ret[j] = grad[j]-proj*trust[j];
    if(ret[j]*grad[j] < 0) ret[j] = 0;
  }
  return ret;
}

inline Real clip(const Real val, const Real ub, const Real lb)
{
  //printf("%f %f\n", ub, lb);
  assert(!std::isnan(val) && !std::isnan(ub) && !std::isnan(lb));
  assert(!std::isinf(val) && !std::isinf(ub) && !std::isinf(lb));
  assert(ub>lb);
  return std::max(std::min(val, ub), lb);
}

inline Uint maxInd(const vector<Real>& pol)
{
  Real Val = -1e9;
  Uint Nbest = 0;
  for (Uint i=0; i<pol.size(); ++i)
      if (pol[i]>Val) { Val = pol[i]; Nbest = i; }
  return Nbest;
}

inline Uint maxInd(const vector<Real>& pol, const Uint start, const Uint N)
{
  Real Val = -1e9;
  Uint Nbest = 0;
  for (Uint i=start; i<start+N; ++i)
      if (pol[i]>Val) { Val = pol[i]; Nbest = i-start; }
  return Nbest;
}

inline Real minAbsValue(const Real v, const Real w)
{
  return std::fabs(v)<std::fabs(w) ? v : w;
}

inline void statsGrad(vector<long double>& sum, vector<long double>& sqr, long double& cnt, const vector<Real>& grad)
{
  assert(sum.size() == grad.size() && sqr.size() == grad.size());
  cnt += 1;
  for (Uint i=0; i<grad.size(); i++) {
    sum[i] += grad[i];
    sqr[i] += grad[i]*grad[i];
  }
}

/*
inline void setVecMean(vector<Real>& vals)
{
   assert(vals.size()>1);
  Real mean = 0;
  for (Uint i=1; i<vals.size(); i++) //assume 0 is empty
    mean += vals[i];
  mean /= (Real)(vals.size()-1);
  for (Uint i=0; i<vals.size(); i++)
    vals[i] = mean;
}
*/
