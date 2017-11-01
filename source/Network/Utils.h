/*
 *  Activations.h
 *  rl
 *
 *  Guido Novati on 04.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <cstring>
#define VEC_WIDTH 32

#if 1
  typedef double nnReal;
  #define MPI_NNVALUE_TYPE MPI_DOUBLE
  //#define EXP_CUT 8 //prevent under/over flow with exponentials
  #define EXP_CUT 4 //prevent under/over flow with exponentials
#else
  #define MPI_NNVALUE_TYPE MPI_FLOAT
  typedef float nnReal;
  #define EXP_CUT 4 //prevent under/over flow with exponentials
#endif

typedef nnReal* __restrict__       const nnOpRet;
typedef const nnReal* __restrict__ const nnOpInp;

static const int simdWidth = VEC_WIDTH/sizeof(nnReal);
static const nnReal nnEPS = std::numeric_limits<float>::epsilon();

#ifndef __CHECK_DIFF
  #define LSTM_PRIME_FAC 1 //input/output gates start closed, forget starts open
#else //else we are testing finite diffs
  #define LSTM_PRIME_FAC 0 //otherwise finite differences are small
  #define PRELU_FAC 1
#endif

inline Uint roundUpSimd(const Uint size)
{
  return std::ceil(size/(Real)simdWidth)*simdWidth;
}

static inline nnReal nnSafeExp(const nnReal val)
{
    return std::exp( std::min((nnReal)8., std::max((nnReal)-16.,val) ) );
}

static inline void Lpenalization(nnReal* const weights, const Uint start, const Uint N, const nnReal lambda)
{
  for (Uint i=start; i<start+N; i++)
  #ifdef NET_L1_PENAL
    if(std::fabs(weights[i])>lambda)
      weights[i] += (weights[i]<0 ? lambda : -lambda);
  #else
    weights[i] -= weights[i]*lambda;
  #endif
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
