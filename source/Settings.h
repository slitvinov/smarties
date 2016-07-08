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
//#define _useOMP_
#ifndef REG
#define REG 5
#endif /* REG */

//#define _scaleR_
#define _BPTT_

//#define _dumpNet_

#define M_POL_O _MM_HINT_NTA
#define M_POL_DS _MM_HINT_NTA
#define M_POL_W _MM_HINT_NTA
#define M_POL_G _MM_HINT_T1

//_MM_HINT_T2 _MM_HINT_T1 _MM_HINT_NTA
#if REG == 0
    #define BCAST  _mm256_broadcast_sd
    #define SET0   _mm256_setzero_pd
    #define SET1   _mm256_set1_pd
    #define ADD    _mm256_add_pd
    #define SUB    _mm256_sub_pd
    #define MUL    _mm256_mul_pd
    #define DIV    _mm256_div_pd
    #define MAX    _mm256_max_pd
    #define MIN    _mm256_min_pd
    #define RCP( arg ) ( _mm256_div_pd ( SET1(1.0) , arg ))
    #define SQRT   _mm256_sqrt_pd
    #define RSQRT( arg ) RCP( _mm256_sqrt_pd( arg ) )
    #define LOAD   _mm256_load_pd
    #define STORE  _mm256_store_pd
    #define STREAM _mm256_stream_pd
    #define FETCH(a,b);  _mm_prefetch( a,b );
    #define SIMD 4
    #define M_PF_G 16
    #define M_PF_O 8
    #define M_PF_W 8
    #define M_PF_DS 8
    typedef double Real;
    typedef __m256d vec;
#elif REG == 1 //#define multiply( f1, f2 ) ( f1 * f2 )
    #define BCAST  _mm256_broadcast_ss
    #define SET0   _mm256_setzero_ps
    #define SET1   _mm256_set1_ps
    #define ADD    _mm256_add_ps
    #define SUB    _mm256_sub_ps
    #define MUL    _mm256_mul_ps
    #define DIV    _mm256_div_ps
    #define MAX    _mm256_max_ps
    #define MIN    _mm256_min_ps
    #define RCP    _mm256_rcp_ps
    #define SQRT   _mm256_sqrt_ps
    #define RSQRT  _mm256_rsqrt_ps
    #define LOAD   _mm256_load_ps
    #define STORE  _mm256_store_ps
    #define STREAM _mm256_stream_ps
    #define FETCH(a,b);  _mm_prefetch( a,b );
    #define SIMD 8
    #define M_PF_G 16
    #define M_PF_O 12
    #define M_PF_W 12
    #define M_PF_DS 12
    typedef float Real;
    typedef __m256 vec;
#elif REG == 2
    #define BCAST  _mm_load1_pd
    #define SET0   _mm_setzero_pd
    #define SET1   _mm_set1_pd
    #define ADD    _mm_add_pd
    #define SUB    _mm_sub_pd
    #define MUL    _mm_mul_pd
    #define DIV    _mm_div_pd
    #define MAX    _mm_max_pd
    #define MIN    _mm_min_pd
    #define RCP( arg ) ( _mm_div_pd ( SET1(1.0) , arg ))
    #define SQRT   _mm_sqrt_pd
    #define RSQRT( arg ) RCP( _mm_sqrt_pd( arg ) )
    #define LOAD   _mm_load_pd
    #define STORE  _mm_store_pd
    #define STREAM _mm_stream_pd
    #define FETCH(a,b);  _mm_prefetch( a,b );
    #define SIMD 2
    #define M_PF_G 4
    #define M_PF_O 4
    #define M_PF_W 4
    #define M_PF_DS 4
    typedef double Real;
    typedef __m128d vec;
#elif REG == 3
    #define BCAST  _mm_load1_ps
    #define SET0   _mm_setzero_ps
    #define SET1   _mm_set1_ps
    #define ADD    _mm_add_ps
    #define SUB    _mm_sub_ps
    #define MUL    _mm_mul_ps
    #define DIV    _mm_div_ps
    #define MAX    _mm_max_ps
    #define MIN    _mm_min_ps
    #define RCP    _mm_rcp_ps
    #define SQRT   _mm_sqrt_ps
    #define RSQRT  _mm_rsqrt_ps
    #define LOAD   _mm_load_ps
    #define STORE  _mm_store_ps
    #define STREAM _mm_stream_ps
    #define FETCH(a,b);  _mm_prefetch( a,b );
    #define SIMD 4
    #define M_PF_G 8
    #define M_PF_O 8
    #define M_PF_W 8
    #define M_PF_DS 8
    typedef float Real;
    typedef __m128 vec;
#else
    #define SIMD 1
    typedef double Real;
#endif

#define ALLOC 32

typedef volatile Real* ompReal;

struct Settings
{
    
    Settings() :
    saveFreq(1e3), randSeed(0), rewardType(0), senses(0), nAgents(1), nSlaves(1), nThreads(-1),
    nnInputs(-1), nnOutputs(-1), nnLayer1(32), nnLayer2(32), nnLayer3(0), nnLayer4(0), nnLayer5(0),
    nnType(1), dqnAppendS(0), dqnBatch(1), bTrain(1), lRate(.0001), greedyEps(.1), gamma(.9),
    lambda(0), goalDY(0), nnPdrop(0), nnLambda(0), dqnUpdateC(1000.), learner((string)"NFQ"),
    restart((string)"policy"), configFile((string)"factory"), prefix((string)"./"),
    samplesFile((string)"../obs_master.txt")
    {}
              
    int saveFreq, randSeed, rewardType, senses, nAgents, nSlaves, nThreads, nnInputs, nnOutputs, nnLayer1, nnLayer2, nnLayer3, nnLayer4, nnLayer5, nnType, dqnAppendS, dqnBatch, bTrain;
	Real lRate, greedyEps, gamma, lambda, goalDY, nnPdrop, nnLambda, dqnUpdateC;
    string learner, restart, configFile, prefix, samplesFile;
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
