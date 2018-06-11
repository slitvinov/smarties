//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the “CC BY-SA 4.0” license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

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
#include <iomanip>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdio.h>

#include <omp.h>
#include <mpi.h>

// Data format for storage in memory buffer. Switch to float for example for
// Atari where the memory buffer is in the order of GBs.
typedef double memReal;
//typedef float memReal;

// Learn rate for the exponential average of the gradient's second moment
// Used to learn the scale for the pre-backprop gradient clipping.
// (currently set to be the same as Adam's second moment learn rate)
#define CLIP_LEARNR 1e-3

// hint to reserve memory for the network workspaces, can be breached
#define MAX_SEQ_LEN 1000

#define FREQ_BACKUP 1000000
// Default number of second moments to clip the pre-backprop gradient:
// Can be changed inside each learning algo by overwriting default arg of
// Approximator::initializeNetwork function
#define STD_GRADCUT 0

// Learn rate of moving stdev and mean of states. If <=0 averaging switched off
#define LEARN_STSCALE 1

// Switch between log(1+exp(x)) and (x+sqrt(x*x+1)/2 as mapping to R^+ for
// policies, advantages, and all math objects that require pos def net outputs
//#define CHEAP_SOFTPLUS

// Switch between network computing \sigma or \Sigma.
#define EXTRACT_COVAR

// Switch between \sigma in (0 1) or (0 inf).
#define UNBND_VAR

#define PRFL_DMPFRQ 50 // regulates how frequently print profiler info

// truncate gaussian dist from -4 to 4, resamples once every ~15787 times.
#define NORMDIST_MAX 3
// bound of pol mean for bounded act. spaces (ie tanh(+/- 8))
#define BOUNDACT_MAX 8

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
//#define NOISY_INPUT 0.001

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

inline void real2SS(ostringstream&B,const Real V,const int W, const bool bPos)
{
  B<<" "<<std::setw(W);
  if(std::fabs(V)>= 1e4) B << std::setprecision(std::max(W-7+bPos,0));
  else
  if(std::fabs(V)>= 1e3) B << std::setprecision(std::max(W-6+bPos,0));
  else
  if(std::fabs(V)>= 1e2) B << std::setprecision(std::max(W-5+bPos,0));
  else
  if(std::fabs(V)>= 1e1) B << std::setprecision(std::max(W-4+bPos,0));
  else
                         B << std::setprecision(std::max(W-3+bPos,0));
  B<<std::fixed<<V;
}

inline bool isZero(const Real vals)
{
  return std::fabs(vals) < std::numeric_limits<Real>::epsilon();
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
