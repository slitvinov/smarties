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
#include <fstream>
#include <iostream>

#include <omp.h>
#include <mpi.h>

#define ACER_BOUNDED //increased safety, TODO move to makefile
#ifdef ACER_BOUNDED
//FOR CONTINUOUS ACTIONS RACER:
  #define ACER_MAX_PREC 1e6
  //FOR DISCRETE ACTIONS DACER:
  #define ACER_MIN_PROB 0.001
#else
  #define ACER_MAX_PREC 1e9
  #define ACER_MIN_PROB 0
#endif

//de facto unused:
#define NORMDIST_MAX 2 //truncated normal distribution range
#define BOUNDACT_MAX 8 //for bounded action spaces: range (ie. tanh(8))

#define MAX_UNROLL_AFTER 2000
#define ACER_CONST_PREC 50 //uniform precision (1/std^2) in case of ACER_SAFE

// number of previous time steps to include in back-prop through time:
#define MAX_UNROLL_BFORE 20
//clip gradients of network outputs if more than # standard deviations from 0:
#define ACER_GRAD_CUT 10
#define MAX_IMPW (Real)100

#ifdef IMPORTSAMPLE
  #define importanceSampling
#endif

//#define NET_L1_PENAL
//#define INTEGRATEANDFIREMODEL
//#define INTEGRATEANDFIRESHARED //if IaF parameters are shared by all neurons
//#define _dumpNet_
#define FULLTASKING

typedef unsigned Uint;

//typedef long double Real;
//#define MPI_VALUE_TYPE MPI_LONG_DOUBLE
typedef double Real;
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

inline Real safeExp(const Real val)
{
  return std::exp( std::min((Real)16, std::max((Real)-32,val) ) );
}

inline vector<Uint> count_indices(const vector<Uint> outs)
{
  vector<Uint> ret(outs.size(), 0); //index 0 is 0
  for(Uint i=1; i<outs.size(); i++) ret[i] = ret[i-1] + outs[i-1];
  return ret;
}
