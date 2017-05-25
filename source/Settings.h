/*
 *  Settings.h
 *  rl
 *
 *  Created by Dmitry Alexeev and extended by Guido Novati on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once

using namespace std;

//#include <cstddef>
#include <utility>
//#define __posDef_layers_
#include <string>
#include <random>
#include <vector>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <immintrin.h>

#include <omp.h>
#include <mpi.h>
//#define __vec_width__ 32
#define __vec_width__ 8
//#define _scaleR_
#define _BPTT_
#define _dumpNet_

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
inline string printVec(const vector<T> vals)
{
  std::ostringstream o;
  for (int i=0; i<vals.size(); i++) o << " " << vals[i];
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
	for (unsigned int i=1; i<vals.size(); i++) //assume 0 is empty
		mean += vals[i];
	mean /= (Real)(vals.size()-1);
	for (unsigned int i=0; i<vals.size(); i++)
		vals[i] = mean;
}

inline void statsVector(vector<vector<Real>>& sum, vector<vector<Real>>& sqr,
  vector<Real>& cnt)
{
   assert(sum.size()>1);
  assert(sum.size() == cnt.size() && sqr.size() == cnt.size());

  for (unsigned int i=0; i<sum[0].size(); i++)
    sum[0][i] = sqr[0][i] = 0;
  cnt[0] = 0;

  for (unsigned int i=1; i<sum.size(); i++) {
    cnt[0] += cnt[i]; cnt[i] = 0;
    for (unsigned int j=0; j<sum[0].size(); j++)
    {
      sum[0][j] += sum[i][j]; sum[i][j] = 0;
      sqr[0][j] += sqr[i][j]; sqr[i][j] = 0;
    }
  }
  cnt[0] = std::max(2.2e-16, cnt[0]);
  for (unsigned int j=0; j<sum[0].size(); j++)
  {
    sqr[0][j] = std::sqrt((sqr[0][j]-sum[0][j]*sum[0][j]/cnt[0])/cnt[0]);
    sum[0][j] /= cnt[0];
  }
}

inline void statsGrad(vector<Real>& sum, vector<Real>& sqr, Real& cnt, vector<Real> grad)
{
  assert(sum.size() == grad.size() && sqr.size() == grad.size());
  cnt += 1;
  for (unsigned int i=0; i<grad.size(); i++) {
    sum[i] += grad[i];
    sqr[i] += grad[i]*grad[i];
  }
}

inline void Lpenalization(Real* const weights, const int start, const int N, const Real lambda)
{
  for (int i=start; i<start+N; i++) weights[i]+= (weights[i]<0 ? lambda : -lambda);
  //for (int i=start; i<start+N; i++) weights[i]-= weights[i]*lambda;
}

struct Settings
{
    Settings() {}
    int saveFreq = 1e3;
    int randSeed = 0;
    int rewardType = 0;
    int senses = 0;
    int nAgents = -1;
    int nSlaves = -1;
    int nThreads = -1;
    int nnInputs = -1;
    int nnOutputs = -1;
    int nnLayer1 = -1;
    int nnLayer2 = -1;
    int nnLayer3 = -1;
    int nnLayer4 = -1;
    int nnLayer5 = -1;
    int dqnAppendS = 0;
    int dqnBatch = 1;
    int maxSeqLen = 1000;
    int minSeqLen = 3;
    int maxTotSeqNum = 5000;
    int nMasters = 1;
    int isLauncher = 1;
    int sockPrefix = 0;
    int separateOutputs = 1;
    Real lRate = 0;
    Real greedyEps = 0.1;
    Real gamma = 0.9;
    Real lambda = 0;
    Real goalDY = 0;
    Real nnPdrop = 0;
    Real nnLambda = 0;
    Real dqnUpdateC = 10000;
    Real epsAnneal = 1e4;
    string learner = "NFQ";
    string restart = "policy";
    string configFile = "factory";
    string prefix = "./";
    string samplesFile = "history.txt";
    string netType = "Feedforward";
    string funcType = "PRelu";
    bool normalizeInput = true;
    bool bIsMaster = true;
    bool bRecurrent = false;
    int bTrain = 1;
    //std::mt19937* gen;
    std::vector<std::mt19937> generators;

    ~Settings()
    {
    	//_dispose_object(gen);
    }

    vector<int> readNetSettingsSize()
    {
      vector<int> ret;
      assert(nnLayer1>0);
      ret.push_back(nnLayer1);
      if (nnLayer2>0) {
        ret.push_back(nnLayer2);
        if (nnLayer3>0) {
          ret.push_back(nnLayer3);
          if (nnLayer4>0) {
            ret.push_back(nnLayer4);
            if (nnLayer5>0) {
              ret.push_back(nnLayer5);
            }
          }
        }
      }
      return ret;
    }
};

namespace ErrorHandling
{
    extern int debugLvl;

#define    die(format, ...) {fprintf(stderr, format, ##__VA_ARGS__); MPI_Abort(MPI_COMM_WORLD, 1);}
#define  error(format, ...) fprintf(stderr, format, ##__VA_ARGS__)

#define   warn(format, ...)	{if (debugLvl > 0) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define  _info(format, ...)	{if (debugLvl > 1) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}

#define  debug(format, ...)	{if (debugLvl > 2) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug1(format, ...)	{if (debugLvl > 3) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug2(format, ...)	{if (debugLvl > 4) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug3(format, ...)	{if (debugLvl > 5) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug4(format, ...)	{if (debugLvl > 6) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug5(format, ...)	{if (debugLvl > 7) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug6(format, ...)	{if (debugLvl > 8) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug7(format, ...)	{if (debugLvl > 9) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug8(format, ...)	{if (debugLvl >10) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
#define debug9(format, ...)	{if (debugLvl >11) fprintf(stderr, format, ##__VA_ARGS__); fflush(0);}
}
