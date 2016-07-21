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

#include <cstddef>
#include <utility>
#include <string>
#include <random>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <immintrin.h>

#include <omp.h>

#define _scaleR_
#define _BPTT_
//#define _dumpNet_

//#define SIMD 1
#define ALLOC 32
typedef double Real;


struct Settings
{
    Settings() :
    saveFreq(1e3), randSeed(0), rewardType(0), senses(0), nAgents(1), nSlaves(1), nThreads(-1),
    nnInputs(-1), nnOutputs(-1), nnLayer1(32), nnLayer2(32), nnLayer3(0), nnLayer4(0), nnLayer5(0),
    nnType(1), dqnAppendS(0), dqnBatch(1), bTrain(1), lRate(.0001), greedyEps(.1), gamma(.9),
    lambda(0), goalDY(0), nnPdrop(0), nnLambda(0), dqnUpdateC(1000.), learner((string)"NFQ"),
    restart((string)"policy"), configFile((string)"factory"), prefix((string)"./"),
    samplesFile((string)"../obs_master.txt"), bSeparateOutputs(false)
    {}

    int saveFreq, randSeed, rewardType, senses, nAgents, nSlaves, nThreads, nnInputs, nnOutputs, nnLayer1, nnLayer2, nnLayer3, nnLayer4, nnLayer5, nnType, dqnAppendS, dqnBatch, bTrain;
	Real lRate, greedyEps, gamma, lambda, goalDY, nnPdrop, nnLambda, dqnUpdateC;
    string learner, restart, configFile, prefix, samplesFile;
    bool bSeparateOutputs;
    mt19937 * gen;
};

namespace ErrorHandling
{
    extern int debugLvl;
    
#define    die(format, ...) fprintf(stderr, format, ##__VA_ARGS__), abort()
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
