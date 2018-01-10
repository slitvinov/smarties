/*
 *  Functions.h
 *  rl
 *
 *  Guido Novati on 04.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */
#pragma once

#include "Functions.h"

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

  Memory(vector<Uint>_sizes, vector<Uint>_bOut): nLayers(_sizes.size()),
  outvals(allocate_vec(_sizes)), sizes(_sizes) {}

  inline void clearOutput() const {
    for(Uint i=0; i<nLayers; i++) {
      const int sizesimd = std::ceil(sizes[i]*sizeof(nnReal)/32.)*32;
      assert(outvals[i] not_eq nullptr && sizes[i]>0);
      std::memset(outvals[i], 0, sizesimd);
    }
  }

  ~Memory() { for(auto& p : outvals) if(p not_eq nullptr) free(p); }
  const Uint nLayers;
  const vector<nnReal*> outvals;
  const vector<Uint> sizes;
};
