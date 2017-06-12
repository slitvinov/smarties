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

#include <random>
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

#define ACER_BOUNDED //increased safety, TODO move to makefile
#ifdef ACER_BOUNDED
//FOR CONTINUOUS ACTIONS RACER:
#define ACER_MAX_PREC 100.
#define ACER_MIN_PREC 1./ACER_MAX_ACT/ACER_MAX_ACT
#define ACER_MAX_ACT 10.
#define ACER_TOL_REW 0.01
#define ACER_TOL_DIAG sqrt(ACER_TOL_REW/ACER_MAX_ACT/ACER_MAX_ACT)
//FOR DISCRETE ACTIONS DACER:
#define ACER_MIN_PROB 0.001
#else
#define ACER_TOL_DIAG 0
#define ACER_MIN_PREC 0
#define ACER_MIN_PROB 0
#endif

#define MAX_UNROLL_AFTER 1000
//#define MAX_UNROLL_AFTER 700
#define MAX_UNROLL_BFORE 16
#define ACER_CONST_PREC 50
#define ACER_GRAD_CUT 1000

//#define NET_L1_PENAL
#define _dumpNet_

typedef double Real;
typedef unsigned Uint;
#define MPI_VALUE_TYPE MPI_DOUBLE

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
