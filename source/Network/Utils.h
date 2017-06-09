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
#define __vec_width__ 8
typedef float nnReal;
static const int simdWidth = __vec_width__/sizeof(nnReal);

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
