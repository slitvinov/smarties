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
  #define ACER_MAX_PREC 400.
  #define ACER_MAX_ACT 10.
  #define ACER_MIN_PREC 4./ACER_MAX_ACT/ACER_MAX_ACT
  #define ACER_TOL_REW 0.01
  #define ACER_TOL_DIAG sqrt(ACER_TOL_REW/ACER_MAX_ACT/ACER_MAX_ACT)
//FOR DISCRETE ACTIONS DACER:
  #define ACER_MIN_PROB 0.001
#else
  #define ACER_TOL_DIAG 0
  #define ACER_MIN_PREC 0
  #define ACER_MIN_PROB 0
#endif
#define ACER_CONST_PREC 50 //uniform precision (1/std^2) in case of ACER_SAFE
#define ACER_LAMBDA 1.0

//#ifndef CUT_TRACES
//  //max number of future time steps for off-policy corrections:
#define MAX_UNROLL_AFTER 2000
//
//#else //lambda 0.5 allows cutting traces
//  #define MAX_UNROLL_AFTER 20 // since (0.5)^20 is ~1e-6 we can truncate?
//  #define ACER_LAMBDA 0.5
//#endif

// number of previous time steps to include in back-prop through time:
#define MAX_UNROLL_BFORE 20
//clip gradients of network outputs if more than # standard deviations from 0:
#define ACER_GRAD_CUT 10

#ifdef IMPORTSAMPLE
  #define importanceSampling
#endif

#define NET_L1_PENAL
//#define INTEGRATEANDFIREMODEL
//#define INTEGRATEANDFIRESHARED //if IaF parameters are shared by all neurons
//#define _dumpNet_
#define FULLTASKING

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
