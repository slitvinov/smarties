//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_FunctionUtilties_h
#define smarties_FunctionUtilties_h

#include "Bund.h"

#include <vector>
#include <cmath> // log, exp, ...
#include <limits>

namespace smarties
{

namespace Utilities
{

inline bool isZero(const Real vals)
{
  return std::fabs(vals) < std::numeric_limits<Real>::epsilon();
}

inline bool nonZero(const Real vals)
{
  return std::fabs(vals) > std::numeric_limits<Real>::epsilon();
}

inline bool isPositive(const Real vals)
{
  return vals > std::numeric_limits<Real>::epsilon();
}

template <typename T>
inline bool isValidValue(const T vals) {
  return ( not std::isnan(vals) ) and ( not std::isinf(vals) );
}

inline Real safeExp(const Real val)
{
  return std::exp( std::min((Real)EXP_CUT, std::max(-(Real)EXP_CUT, val) ) );
}
template <typename T = nnReal>
inline T nnSafeExp(const T val)
{
  return std::exp( std::min( (T)EXP_CUT, std::max( - (T)EXP_CUT, val) ) );
}

inline std::vector<Uint> count_indices(const std::vector<Uint> outs)
{
  std::vector<Uint> ret(outs.size(), 0); //index 0 is 0
  for(Uint i=1; i<outs.size(); i++) ret[i] = ret[i-1] + outs[i-1];
  return ret;
}

template <typename T>
inline T annealRate(const T eta, const Real t, const Real T)
{
  return eta / (1 + (T) t * T);
}

inline Uint roundUpSimd(const Real N)
{
  return std::ceil(N/ARY_WIDTH)*ARY_WIDTH;
}

inline nnReal* allocate_dirty(const Uint _size)
{
  nnReal* ret = nullptr;
  assert(_size > 0);
  posix_memalign((void **) &ret, VEC_WIDTH, roundUpSimd(_size)*sizeof(nnReal));
  return ret;
}

inline nnReal* allocate_ptr(const Uint _size)
{
  nnReal* ret = allocate_dirty(_size);
  memset(ret, 0, roundUpSimd(_size)*sizeof(nnReal) );
  return ret;
}

inline nnReal* allocate_param(const Uint _size, const Real mpiSz)
{
  const Uint extraSize = roundUpSimd(std::ceil(_size / mpiSz)) * mpiSz;
  //printf("requested %u floats of size %lu, allocating %lu bytes\n",
  //  _size, sizeof(nnReal), extraSize*sizeof(nnReal));
  nnReal* ret = allocate_dirty(extraSize);
  memset(ret, 0, extraSize * sizeof(nnReal) );
  return ret;
}


inline std::vector<nnReal*> allocate_vec(std::vector<Uint> _sizes)
{
  std::vector<nnReal*> ret(_sizes.size(), nullptr);
  for(Uint i=0; i<_sizes.size(); i++) ret[i] = allocate_ptr(_sizes[i]);
  return ret;
}

#ifdef CHEAP_SOFTPLUS
  inline Real unbPosMap_func(const Real in)
  {
    return  .5*(in + std::sqrt(1+in*in));
  }
  inline Real unbPosMap_diff(const Real in)
  {
    return .5*(1 + in/std::sqrt(1+in*in));
  }
  inline Real unbPosMap_inverse(Real in)
  {
    if(in<=0) {
      warn("Tried to initialize invalid pos-def mapping. Unless not training this should not be happening. Revise setting explNoise.");
      in = std::numeric_limits<float>::epsilon();
    }
    return (in*in - 0.25)/in;
  }
#else
  inline Real unbPosMap_func(const Real in)
  {
    return std::log(1+safeExp(in));
  }
  inline Real unbPosMap_diff(const Real in)
  {
    return 1/(1+safeExp(-in));
  }
  inline Real unbPosMap_inverse(Real in)
  {
    if(in<=0) {
      warn("Tried to initialize invalid pos-def mapping. Unless not training this should not be happening. Revise setting explNoise.");
      in = std::numeric_limits<float>::epsilon();
    }
    return std::log(safeExp(in)-1);
  }
#endif

inline Real noiseMap_func(const Real val)
{
  #ifdef UNBND_VAR
    return unbPosMap_func(val);
  #else
    return Sigm::_eval(val);
  #endif
}
inline Real noiseMap_diff(const Real val)
{
  #ifdef UNBND_VAR
    return unbPosMap_diff(val);
  #else
    return Sigm::_evalDiff(val);
  #endif
}
inline Real noiseMap_inverse(Real val)
{
  #ifdef UNBND_VAR
    return unbPosMap_inverse(val);
  #else
    if(val<=0 || val>=1) {
      warn("Tried to initialize invalid pos-def mapping. Unless not training this should not be happening. Revise setting explNoise.");
      if(val<=0) val =   std::numeric_limits<float>::epsilon();
      if(val>=1) val = 1-std::numeric_limits<float>::epsilon();
    }
    return Sigm::_inv(val);
  #endif
}

inline Real clip(const Real val, const Real ub, const Real lb)
{
  assert(!std::isnan(val) && !std::isnan(ub) && !std::isnan(lb));
  assert(!std::isinf(val) && !std::isinf(ub) && !std::isinf(lb));
  assert(ub>lb);
  return std::max(std::min(val, ub), lb);
}

inline Rvec sum3Grads(const Rvec& f, const Rvec& g, const Rvec& h)
{
  assert(g.size() == f.size());
  assert(h.size() == f.size());
  Rvec ret(f.size());
  for(Uint i=0; i<f.size(); i++) ret[i] = f[i]+g[i]+h[i];
  return ret;
}

inline Rvec sum2Grads(const Rvec& f, const Rvec& g)
{
  assert(g.size() == f.size());
  Rvec ret(f.size());
  for(Uint i=0; i<f.size(); i++) ret[i] = f[i]+g[i];
  return ret;
}

inline Rvec weightSum2Grads(const Rvec& f, const Rvec& g, const Real W)
{
  assert(g.size() == f.size());
  Rvec ret(f.size());
  for(Uint i=0; i<f.size(); i++) ret[i] = W*f[i]+ (1-W)*g[i];
  return ret;
}

inline Rvec trust_region_update(const Rvec& grad,
  const Rvec& trust, const Uint nA, const Real delta)
{
  assert(grad.size() == trust.size());
  Rvec ret(nA);
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
  for (Uint j=0; j<nA; j++) ret[j] = grad[j]-proj*trust[j];
  return ret;
}

inline Uint maxInd(const Rvec& pol)
{
  Real Val = -1e9;
  Uint Nbest = 0;
  for (Uint i=0; i<pol.size(); ++i)
      if (pol[i]>Val) { Val = pol[i]; Nbest = i; }
  return Nbest;
}

inline Uint maxInd(const Rvec& pol, const Uint start, const Uint N)
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

template <typename T>
void dispose_object(T *& ptr)
{
  if(ptr == nullptr) return;
  delete ptr;
  ptr=nullptr;
}

template <typename T>
void dispose_object(T *const& ptr)
{
  if(ptr == nullptr) return;
  delete ptr;
}


} // end namespace smarties
} // end namespace Utilities
#endif // smarties_FunctionUtilties_h

/*
inline Uint roundUpSimd(const Uint size)
{
  return std::ceil(size/(Real)simdWidth)*simdWidth;
}
static inline nnReal readCutStart(vector<nnReal>& buf)
{
  const Real ret = buf.front();
  buf.erase(buf.begin(),buf.begin()+1);
  assert(!std::isnan(ret) && !std::isinf(ret));
  return ret;
}
static inline nnReal readBuf(vector<nnReal>& buf)
{
  //const Real ret = buf.front();
  //buf.erase(buf.begin(),buf.begin()+1);
  const Real ret = buf.back();
  buf.pop_back();
  assert(!std::isnan(ret) && !std::isinf(ret));
  return ret;
}
static inline void writeBuf(const nnReal weight, vector<nnReal>& buf)
{
  buf.insert(buf.begin(), weight);
}

template <typename T>
inline void _myfree(T *const& ptr)
{
  if(ptr == nullptr) return;
  free(ptr);
}

//template <typename T>
inline void _allocate_quick(nnReal*& ptr, const Uint size)
{
  const Uint sizeSIMD = roundUpSimd(size)*sizeof(nnReal);
  posix_memalign((void **) &ptr, VEC_WIDTH, sizeSIMD);
}

//template <typename T>
inline void _allocate_clean(nnReal*& ptr, const Uint size)
{
  const Uint sizeSIMD = roundUpSimd(size)*sizeof(nnReal);
  posix_memalign((void **) &ptr, VEC_WIDTH, sizeSIMD);
  memset(ptr, 0, sizeSIMD);
}

inline nnReal* init(const Uint N, const nnReal ini)
{
  nnReal* ret;
  _allocate_quick(ret, N);
  for (Uint j=0; j<N; j++) ret[j] = ini;
  return ret;
}

inline nnReal* initClean(const Uint N)
{
  nnReal* ret;
  _allocate_clean(ret, N);
  return ret;
}

inline nnReal* init(const Uint N)
{
  nnReal* ret;
  _allocate_quick(ret, N);
  return ret;
}

inline Rvec circle_region(const Rvec& grad,
  const Rvec& trust, const Uint nact, const Real delta)
{
  assert(grad.size() == trust.size());
  const Uint nA = grad.size();
  Rvec ret(nA);
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
*/
