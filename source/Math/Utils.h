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

inline Real clip(const Real val, const Real ub, const Real lb)
{
  assert(!std::isnan(val) && !std::isnan(ub) && !std::isnan(lb));
  assert(!std::isinf(val) && !std::isinf(ub) && !std::isinf(lb));
  assert(ub>lb);
  return std::max(std::min(val, ub), lb);
}

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

inline vector<Real> weightSum2Grads(const vector<Real>& f,
  const vector<Real>& g, const Real W)
{
  assert(g.size() == f.size());
  vector<Real> ret(f.size());
  for(Uint i=0; i<f.size(); i++) ret[i] = W*f[i]+ (1-W)*g[i];
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
  const Real proj = std::max((Real)0, (dot-delta)/norm);
  //#ifndef NDEBUG
  //if(proj>0) {printf("Hit DKL constraint\n");fflush(0);}
  //else {printf("Not Hit DKL constraint\n");fflush(0);}
  //#endif
  for (Uint j=0; j<nA; j++) {
    ret[j] = grad[j]-proj*trust[j];
    //if(ret[j]*grad[j] < 0) ret[j] = 0;
  }
  return ret;
}

inline vector<Real> circle_region(const vector<Real>& grad,
  const vector<Real>& trust, const Uint nact, const Real delta)
{
  assert(grad.size() == trust.size());
  const Uint nA = grad.size();
  vector<Real> ret(nA);
  Real normKG = 0, normK = 1e-16, normG = 1e-16, dot = 0;
  for(Uint j=0; j<nA; j++) {
    normKG += std::pow(grad[j]+trust[j],2);
    normK += trust[j] * trust[j];
    normG += grad[j] * grad[j];
    dot += trust[j] *  grad[j];
  }
  #if 1
    const Real nG = sqrt(normG)/nact *(sqrt(normK)/nact +dot/sqrt(normG)/nact);
    const Real denom = std::max((Real)1, nG/delta);
    //const Real denom = (1+ nG/delta);
    for(Uint j=0; j<nA; j++) ret[j] = grad[j]/denom;
  #else
    const Real nG = std::sqrt(normKG)/nact;
    const Real denom = std::max((Real)1, nG/delta);
    //const Real denom = (1+ nG/delta);
    const Real numer = std::min((Real)0, (delta-nG)/delta);
    for(Uint j=0; j<nA; j++) ret[j] = (grad[j] + numer*trust[j])/denom;
  #endif
  //printf("KG:%f K:%f G:%f dot:%f denom:%f delta:%f\n",
  //       normKG,normK,normG,dot,denom,delta);
  //const Real nG = std::sqrt(normKG), softclip = delta/(nG+delta);
  //for(Uint j=0; j<nA; j++) ret[j] = (grad[j]+trust[j])*softclip -trust[j];
  return ret;
}

inline void circle_region(vector<Real>& grad, const Real delta, const Uint start, const Uint end)
{
  Real normKG = 0;
  for(Uint j=start; j<end; j++) normKG += std::pow(grad[j],2);
  const Real nG = std::sqrt(normKG), softclip = delta/(nG+delta);
  for(Uint j=start; j<end; j++) grad[j] = grad[j]*softclip;
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
