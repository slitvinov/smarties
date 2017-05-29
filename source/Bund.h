/*
 *  Settings.h
 *  rl
 *
 *  Created by Guido Novati on 02.05.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once

using namespace std;

//#define __posDef_layers_
#include <vector>
#include <cassert>
#include <sstream>
#include <cstring>
#include <utility>
#include <limits>
#include <cmath>
#include <immintrin.h>

#include <omp.h>
#include <mpi.h>
#define __vec_width__ 32
//#define __vec_width__ 8

#define _BPTT_
#define _dumpNet_

typedef double Real;
typedef unsigned Uint;
#define MPI_VALUE_TYPE MPI_DOUBLE
static const Uint simdWidth = __vec_width__/sizeof(Real);

template <typename T>
inline void _myfree(T *const& ptr)
{
  if(ptr == nullptr) return;
  free(ptr);
}

template <typename T>
inline void _allocateQuick(T *const& ptr, const Uint size)
{
    const Uint sizeSIMD=std::ceil(size/(Real)simdWidth)*simdWidth*sizeof(Real);
    posix_memalign((void **)& ptr, __vec_width__, sizeSIMD);
}

template <typename T>
inline void _allocateClean(T *const& ptr, const Uint size)
{
    const Uint sizeSIMD=std::ceil(size/(Real)simdWidth)*simdWidth*sizeof(Real);
    posix_memalign((void **)& ptr, __vec_width__, sizeSIMD);
    memset(ptr, 0, sizeSIMD);
}

template <typename T>
void _dispose_object(T *& ptr)
{
    if(ptr == nullptr) return;
    delete ptr;
    ptr=nullptr;
}

template <typename T>
void _dispose_object(T *const& ptr)
{
    if(ptr == nullptr) return;
    delete ptr;
}

inline Real* init(const Uint N, const Real ini)
{
  Real* ret;
  _allocateQuick(ret, N);
  for (Uint j=0; j<N; j++) ret[j] = ini;
  return ret;
}

inline Real* initClean(const Uint N)
{
  Real* ret;
  _allocateClean(ret, N);
  return ret;
}

inline Real* init(const Uint N)
{
  Real* ret;
  _allocateQuick(ret, N);
  return ret;
}

inline Real safeExp(const Real val)
{
    return std::exp( std::min(9., std::max(-32.,val) ) );
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

inline Real clip(const Real val, const Real ub, const Real lb)
{
  assert(!isnan(val));
  assert(!isinf(val));
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

inline Real minAbsValue(const Real v, const Real w)
{
  return std::fabs(v)<std::fabs(w) ? v : w;
}

template <typename T>
inline string print(const vector<T> vals)
{
  std::ostringstream o;
  if(!vals.size()) return o.str();
  for (Uint i=0; i<vals.size()-1; i++) o << vals[i] << " ";
  o << vals[vals.size()-1];
  return o.str();
}

inline bool nonZero(const Real vals)
{
  return std::fabs(vals) > std::numeric_limits<Real>::epsilon();
}

inline bool positive(const Real vals)
{
  return vals > std::numeric_limits<Real>::epsilon();
}

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

inline void statsVector(vector<vector<Real>>& sum, vector<vector<Real>>& sqr,
  vector<Real>& cnt)
{
   assert(sum.size()>1);
  assert(sum.size() == cnt.size() && sqr.size() == cnt.size());

  for (Uint i=0; i<sum[0].size(); i++)
    sum[0][i] = sqr[0][i] = 0;
  cnt[0] = 0;

  for (Uint i=1; i<sum.size(); i++) {
    cnt[0] += cnt[i]; cnt[i] = 0;
    for (Uint j=0; j<sum[0].size(); j++)
    {
      sum[0][j] += sum[i][j]; sum[i][j] = 0;
      sqr[0][j] += sqr[i][j]; sqr[i][j] = 0;
    }
  }
  cnt[0] = std::max(2.2e-16, cnt[0]);
  for (Uint j=0; j<sum[0].size(); j++)
  {
    sqr[0][j] = std::sqrt((sqr[0][j]-sum[0][j]*sum[0][j]/cnt[0])/cnt[0]);
    sum[0][j] /= cnt[0];
  }
}

inline void statsGrad(vector<Real>& sum, vector<Real>& sqr, Real& cnt, vector<Real> grad)
{
  assert(sum.size() == grad.size() && sqr.size() == grad.size());
  cnt += 1;
  for (Uint i=0; i<grad.size(); i++) {
    sum[i] += grad[i];
    sqr[i] += grad[i]*grad[i];
  }
}

inline void Lpenalization(Real* const weights, const Uint start, const Uint N, const Real lambda)
{
  for (Uint i=start; i<start+N; i++) weights[i]+= (weights[i]<0 ? lambda : -lambda);
  //for (int i=start; i<start+N; i++) weights[i]-= weights[i]*lambda;
}
