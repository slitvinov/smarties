/*
 *  Functions.h
 *  rl
 *
 *  Guido Novati on 04.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once


struct MemoryManager
{
  const int nThreads;
  vector<Parameters*> gradients;
  vector<Parameters*> gradients_cuda;
  vector<vector<Activation*>> activations;
  vector<vector<Activation*>> activations_cuda;
  vector<pair<nnReal*, Uint>> workspaces_cudnn;
  vector<cublasHandle_t> handles_cublas;
  vector<cudnnHandle_t> handles_cudnn;
  vector<cudaStream_t> streams_cuda;
};
