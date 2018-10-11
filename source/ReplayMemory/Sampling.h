//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#pragma once

#include "Sequences.h"
#include <atomic>

class MemoryBuffer;

class Sampling
{
 protected:
  std::vector<std::mt19937>& gens;
  MemoryBuffer* const RM;
  const std::vector<Sequence*>& Set;
  long nSequences() const;
  long nTransitions() const;
  void setMinMaxProb(const Real maxP, const Real minP);

 public:
  Sampling(const Settings& S, MemoryBuffer*const R);
  virtual void sample(vector<Uint>& seq, vector<Uint>& obs) = 0;
  virtual void prepare(std::atomic<bool>& needs_pass) = 0;
  void IDtoSeqStep(std::vector<Uint>& seq, std::vector<Uint>& obs,
                  const std::vector<Uint>& ret, const Uint nSeqs);
};

class TSample_uniform : public Sampling
{
 public:
  TSample_uniform(const Settings& S, MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
};

class TSample_impLen : public Sampling
{
  std::discrete_distribution<Uint> dist;
 public:
  TSample_impLen(const Settings& S, MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
};

class SSample_uniform : public Sampling
{
 public:
  SSample_uniform(const Settings& S, MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
};

class TSample_shuffle : public Sampling
{
  std::vector<std::pair<unsigned, unsigned>> samples;
 public:
  TSample_shuffle(const Settings& S, MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
};

#ifdef PRIORITIZED_ER

class TSample_impRank : public Sampling
{
  int stepSinceISWeep = 0;
  std::discrete_distribution<Uint> distObs;
 public:
  TSample_impRank(const Settings& S, MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
};

class TSample_impErr : public Sampling
{
  int stepSinceISWeep = 0;
  std::discrete_distribution<Uint> distObs;
 public:
  TSample_impErr(const Settings& S, MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare(std::atomic<bool>& needs_pass) override;
};

#endif
