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

// Data format for storage in memory buffer. Switch to float for example for
// Atari where the memory buffer is in the order of GBs.
typedef double memReal;
//typedef float memReal;

// If this is uncommented, action importance sampling ratio is defined by
// [ pi(a|s) / mu(a|s) ]^f(nA) where f(nA) is function of a dimensionality
// In ACER's paper f(nA) = 1/nA. For Racer leads to bad perf.
// Tested also 1/sqrt(nA) and 1/cbrt(nA) then gave up.
//#define RACER_ACERTRICK

// Learn rate for the exponential average of the gradient's second moment
// Used to learn the scale for the pre-backprop gradient clipping.
// (currently set to be the same as Adam's second moment learn rate)
#define CLIP_LEARNR 1e-3

// Default number of second moments to clip the pre-backprop gradient:
// Can be changed inside each learning algo by overwriting default arg of
// Approximator::initializeNetwork function
#define STD_GRADCUT 5

// Anneal rate for the learning rate, should be moved to settings
#define ANNEAL_RATE 5e-7
#define ANNEAL_LEARNR // Anneal also network learn rate

#define PRFL_DMPFRQ 50 // regulates how frequently print profiler info

#define NORMDIST_MAX 2 //truncated normal distribution range
#define BOUNDACT_MAX 4 //limit range of pol. for bounded act. spaces ie. tanh(4)

// Optional constant stdev in case of Acer:
#define ACER_CONST_STDEV 0.3

// number of previous time steps to include in back-prop through time:
#define MAX_UNROLL_BFORE 20

//#define NET_L1_PENAL // else employ L2 penal defined by Settings::nnLambda
//#define _dumpNet_
#define SAFE_ADAM // prevent rare gradient blow ups?

// Sample white Gaussian noise and add it to state vector before input to net
// This has been found to help in case of dramatic dearth of data
// The noise stdev for state s_t is = ($NOISY_INPUT) * || s_{t-1} - s_{t+1} ||
//#define NOISY_INPUT 0.01

typedef unsigned Uint;

#if 0
typedef long double Real;
#define MPI_VALUE_TYPE MPI_LONG_DOUBLE
#else
typedef double Real;
#define MPI_VALUE_TYPE MPI_DOUBLE
#endif
typedef vector<Real> Rvec;
typedef vector<long double> LDvec;

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
