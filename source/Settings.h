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

struct Settings
{
    Settings() : saveFreq(1e3), randSeed(0), rewardType(0), senses(0),
    nAgents(1), nSlaves(1), nThreads(-1), nnInputs(-1), nnOutputs(-1),
    nnLayer1(32), nnLayer2(32), nnLayer3(0), nnLayer4(0), nnLayer5(0), nnType(1),
    dqnAppendS(0), dqnBatch(1), bTrain(1), maxSeqLen(200), minSeqLen(5),
    maxTotSeqNum(5000), nMasters(1), isLauncher(1),
    lRate(.0001), greedyEps(.1), gamma(.9), lambda(0), goalDY(0), nnPdrop(0),
    nnLambda(0), dqnUpdateC(1000.), epsAnneal(1e4), learner((string)"NFQ"),
    restart((string)"policy"), configFile((string)"factory"), prefix((string)"./"),
    samplesFile((string)"../obs_master.txt"), bSeparateOutputs(false),
    nnTypeInput(true), bIsMaster(true)
    {}

    int saveFreq, randSeed, rewardType, senses, nAgents, nSlaves, nThreads;
    int nnInputs, nnOutputs, nnLayer1, nnLayer2, nnLayer3, nnLayer4, nnLayer5;
    int nnType, dqnAppendS, dqnBatch, bTrain, maxSeqLen, minSeqLen;
    int maxTotSeqNum, nMasters, isLauncher, sockPrefix;
    Real lRate, greedyEps, gamma, lambda, goalDY, nnPdrop, nnLambda, dqnUpdateC, epsAnneal;
    string learner, restart, configFile, prefix, samplesFile;
    bool bSeparateOutputs, nnTypeInput, bIsMaster;
    //std::mt19937* gen;
    std::vector<std::mt19937> generators;

    ~Settings()
    {
    	//_dispose_object(gen);
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
