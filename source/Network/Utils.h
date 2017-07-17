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
#define __vec_width__ 32
#if 0
typedef double nnReal;
#define MPI_NNVALUE_TYPE MPI_DOUBLE
#define EXP_CUT 8
#else
#define MPI_NNVALUE_TYPE MPI_FLOAT
typedef float nnReal;
#define EXP_CUT 4
#endif
static const int simdWidth = __vec_width__/sizeof(nnReal);
typedef nnReal*__restrict__ const				nnOpRet;
typedef const nnReal*__restrict__ const nnOpInp;

static inline void Lpenalization(nnReal* const weights,
	const Uint start, const Uint N, const nnReal lambda)
{
	for (Uint i=start; i<start+N; i++)
	#ifdef NET_L1_PENAL
		weights[i] += (weights[i]<0 ? lambda : -lambda);
	#else
		weights[i] -= weights[i]*lambda;
	#endif
}

template <typename T>
inline void _myfree(T *const& ptr)
{
  if(ptr == nullptr) return;
  free(ptr);
}

template <typename T>
inline void _allocateQuick(T *const& ptr, const Uint size)
{
    const Uint sizeSIMD =
			std::ceil(size/(nnReal)simdWidth) * simdWidth*sizeof(nnReal);
    posix_memalign((void **)& ptr, __vec_width__, sizeSIMD);
}

template <typename T>
inline void _allocateClean(T *const& ptr, const Uint size)
{
	const Uint sizeSIMD =
		std::ceil(size/(nnReal)simdWidth) * simdWidth*sizeof(nnReal);
    posix_memalign((void **)& ptr, __vec_width__, sizeSIMD);
    memset(ptr, 0, sizeSIMD);
}

inline nnReal* init(const Uint N, const nnReal ini)
{
  nnReal* ret;
  _allocateQuick(ret, N);
  for (Uint j=0; j<N; j++) ret[j] = ini;
  return ret;
}

inline nnReal* initClean(const Uint N)
{
  nnReal* ret;
  _allocateClean(ret, N);
  return ret;
}

inline nnReal* init(const Uint N)
{
  nnReal* ret;
  _allocateQuick(ret, N);
  return ret;
}
