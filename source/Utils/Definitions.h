//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Definitions_h
#define smarties_Definitions_h

#include <vector>
#include <array>

namespace smarties
{

typedef unsigned Uint;
////////////////////////////////////////////////////////////////////////////////
#if 1 // MAIN CODE PRECISION
using Real = double;
#define MPI_VALUE_TYPE MPI_DOUBLE
#else
using Real = float;
#define MPI_VALUE_TYPE MPI_FLOAT
#endif
///////////////////////////////////////////////////////////////////////////////
#ifndef SINGLE_PREC // NETWORK PRECISION
  #define gemv cblas_dgemv
  #define gemm cblas_dgemm
  using nnReal = double;
  #define MPI_NNVALUE_TYPE MPI_DOUBLE
  #define EXP_CUT 16 //prevent under/over flow with exponentials
#else
  #define gemv cblas_sgemv
  #define gemm cblas_sgemm
  #define MPI_NNVALUE_TYPE MPI_FLOAT
  using nnReal = float;
  #define EXP_CUT 8 //prevent under/over flow with exponentials
#endif
////////////////////////////////////////////////////////////////////////////////
// Data format for storage in memory buffer. Switch to float for example for
// Atari where the memory buffer is in the order of GBs.
#ifndef SINGLE_PREC
using memReal = double;
using Fval = double;
#define MPI_Fval MPI_DOUBLE
#else
using memReal = float;
using Fval = float;
#define MPI_Fval MPI_FLOAT
#endif

typedef std::vector<Fval> Fvec;
typedef std::vector<Real> Rvec;
typedef std::vector<long double> LDvec;

struct Conv2D_Descriptor
{
  Uint inpFeatures, inpY, inpX; //input image: channels, x:width, y:height:
  Uint outFeatures, outY, outX; // output image
  Uint filterx, filtery; // tot size : inpFeatures*outFeatures*filterx*filtery
  Uint stridex, stridey;
  Uint paddinx, paddiny;
};

#endif
