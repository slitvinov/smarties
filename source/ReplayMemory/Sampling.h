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
  const MemoryBuffer* const RM;
  const std::vector<Sequence*>& Set;
  long nSequences() const;
  long nTransitions() const;

 public:
  Sampling(const Settings& S, const MemoryBuffer*const R);
  virtual void sample(vector<Uint>& seq, vector<Uint>& obs) = 0;
  virtual void prepare() = 0;
};

class TSample_uniform : public Sampling
{
 public:
  TSample_uniform(const Settings& S, const MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare() override;
};

class TSample_impLen : public Sampling
{
  std::discrete_distribution<Uint> dist;
 public:
  TSample_impLen(const Settings& S, const MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare() override;
};

class SSample_uniform : public Sampling
{
 public:
  SSample_uniform(const Settings& S, const MemoryBuffer*const R);
  void sample(vector<Uint>& seq, vector<Uint>& obs) override;
  void prepare() override;
};
