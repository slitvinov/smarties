/*
 *  Settings.h
 *  rl
 *
 *  Created by Dmitry Alexeev and extended by Guido Novati on 02.05.13.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include <string>
using namespace std;

#include "ErrorHandling.h"
#include <cstddef>
#include <utility>
#include <vector>
#include <immintrin.h>
//#include <omp.h>
//#define _useOMP_
#ifndef REG
#define REG 2
#endif /* REG */

//#define _scaleR_
#define _BPTT_
//#define _Priority_
#define _dumpNet_

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
extern struct Settings
{
    Settings() : configFile((string)"factory"),
    dt(0.01), endTime(1e9), gamma(0.95), greedyEps(0.01),
    prefix((string)"res/"), lRate(0.1), lambda(0.0), randSeed(0),
    restart((string)"res/policy"), saveFreq(1000),
    nnEta(0.001), nnAlpha(0.5), nnLambda(0.0), nnPdrop(0.0),
    AL_fac(0.0), nnLayer1(32), nnLayer2(16),
    nnLayer3( 0), nnMemory1(0), nnMemory2(0), nnMemory3(0),
    EndR(-9.99), bTrain(false), learner ((string)"NFQ"), approx ((string)"NN") {}
              
	int    saveFreq;
	int    videoFreq;
    int    rewardType;
    Real goalDY;
    
    int nAgents, nSlaves;
    
	string configFile;
	Real dt;
	Real endTime;
	int    randSeed;
	
    Real EndR;
	Real lRate;
	Real greedyEps;
	Real gamma;
	Real lambda;
	string restart;
	
	Real nnEta;
	Real nnAlpha;
    Real nnPdrop;
    Real nnLambda;
	int    nnLayer1;
	int    nnLayer2;
    int    nnLayer3;
    int    nnLayer4;
    int    nnLayer5;
    int    nnMemory1;
    int    nnMemory2;
    int    nnMemory3;
    int    nnMemory4;
    int    nnMemory5;
    int    nnOuts;
    
    Real  AL_fac;
    string learner;
    string approx;
    
	bool best;
    int bTrain;
	bool immortal;
	string prefix;
	
} settings;
