//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryBuffer.h"
#include <iterator>
#include <algorithm>

namespace smarties
{

MemoryBuffer::MemoryBuffer(MDPdescriptor&M_, Settings&S_, DistributionInfo&D_):
 MDP(M_), settings(S_), distrib(D_), sampler( prepareSampler(this, S_, D_) )
{
  Set.reserve(settings.maxTotObsNum);
}

void MemoryBuffer::save(const std::string base, const Uint nStep, const bool bBackup)
{
  const auto write2file = [&] (FILE * wFile) {
    std::vector<double> V = std::vector<double>(mean.begin(), mean.end());
    fwrite(V.data(), sizeof(double), V.size(), wFile);
    V = std::vector<double>(invstd.begin(), invstd.end());
    fwrite(V.data(), sizeof(double), V.size(), wFile);
    V = std::vector<double>(std.begin(), std.end());
    fwrite(V.data(), sizeof(double), V.size(), wFile);
    V.resize(2); V[0] = stddev_reward; V[1] = invstd_reward;
    fwrite(V.data(), sizeof(double), 2, wFile);
  }

  FILE * wFile = fopen((base+"scaling.raw").c_str(), "wb");
  write2file(wFile); fflush(wFile); fclose(wFile);

  if(bBackup) {
    std::ostringstream S; S<<std::setw(9)<<std::setfill('0')<<nStep;
    wFile = fopen((base+"scaling_"+S.str()+".raw").c_str(), "wb");
    write2file(wFile); fflush(wFile); fclose(wFile);
  }
}

void MemoryBuffer::restart(const std::string base)
{
  {
    FILE * wFile = fopen((base+"scaling.raw").c_str(), "rb");
    if(wFile == NULL) {
      printf("Parameters restart file %s not found.\n", (base+".raw").c_str());
      return;
    } else {
      printf("Restarting from file %s.\n", (base+"scaling.raw").c_str());
      fflush(0);
    }

    const Uint dimS = MDP.dimStateObserved; assert(mean.size() == dimS);
    std::vector<double> V(dimS);
    size_t size1 = fread(V.data(), sizeof(double), dimS, wFile);
    mean   = std::vector<nnReal>(V.begin(), V.end());
    size_t size2 = fread(V.data(), sizeof(double), dimS, wFile);
    invstd = std::vector<nnReal>(V.begin(), V.end());
    size_t size3 = fread(V.data(), sizeof(double), dimS, wFile);
    std    = std::vector<nnReal>(V.begin(), V.end());
    V.resize(2);
    size_t size4 = fread(V.data(), sizeof(double),    2, wFile);
    stddev_reward = V[0]; invstd_reward = V[1];
    fclose(wFile);
    if (size1!=dimS || size2!=dimS || size3!=dimS || size4!=2)
      _die("Mismatch in restarted file %s.", (base+"_scaling.raw").c_str());
  }
}

void MemoryBuffer::clearAll()
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  //delete already-used trajectories
  for(auto& S: Set) Utilities::dispose_object(S);

  Set.clear(); //clear trajectories used for learning
  nTransitions = 0;
  nSequences = 0;
  needs_pass = true;
}

void MemoryBuffer::sample(std::vector<Uint>& seq, std::vector<Uint>& obs)
{
  sampler->sample(seq, obs);
  // remember which episodes were just sampled:
  sampled = seq;
  std::sort(sampled.begin(), sampled.end());
  sampled.erase(std::unique(sampled.begin(), sampled.end()), sampled.end());

  for(Uint i=0; i<seq.size(); i++) Set[seq[i]]->setSampled(obs[i]);
}

void MemoryBuffer::removeSequence(const Uint ind)
{
  assert(readNSeq()>0);
  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert(nTransitions >= Set[ind]->ndata());
  assert(Set[ind] not_eq nullptr);
  nSequences--;
  needs_pass = true;
  nTransitions -= Set[ind]->ndata();
  std::swap(Set[ind], Set.back());
  Utilities::dispose_object(Set.back());
  Set.pop_back();
  assert(nSequences == (long) Set.size());
}
void MemoryBuffer::pushBackSequence(Sequence*const seq)
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  assert( readNSeq() == (long) Set.size() and seq not_eq nullptr);
  const auto ind = Set.size();
  Set.push_back(seq);
  Set[ind]->prefix = ind>0? Set[ind-1]->prefix +Set[ind-1]->ndata() : 0;
  nTransitions += seq->ndata();
  needs_pass = true;
  nSequences++;
  assert( readNSeq() == (long) Set.size());
}

void MemoryBuffer::initialize()
{
  // All sequences obtained before this point should share the same time stamp
  for(Uint i=0;i<Set.size();i++) Set[i]->ID = nSeenSequences.load();

  needs_pass = true;
  sampler->prepare(needs_pass);
}

MemoryBuffer::~MemoryBuffer()
{
  for(auto & S : Set) Utilities::dispose_object(S);
}

void MemoryBuffer::checkNData()
{
  #ifndef NDEBUG
    Uint cntSamp = 0;
    for(Uint i=0; i<Set.size(); i++) {
      assert(Set[i] not_eq nullptr);
      cntSamp += Set[i]->ndata();
    }
    assert(cntSamp==nTransitions);
    assert(nSequences==(long)Set.size());
  #endif
}

std::unique_ptr<Sampling> MemoryBuffer::prepareSampler(MemoryBuffer* const R, Settings&S_, DistributionInfo&D_)
{
  std::unique_ptr<Sampling> ret = nullptr;

  if(S.dataSamplingAlgo == "uniform") ret = std::make_unique<Sample_uniform>(
    D_.generators, R, S_.bSampleSequences);

  if(S.dataSamplingAlgo == "impLen")  ret = std::make_unique<Sample_impLen>(
    D_.generators, R, S_.bSampleSequences);

  if(S.dataSamplingAlgo == "shuffle") {
    ret = std::make_unique<TSample_shuffle>(
      D_.generators, R, S_.bSampleSequences);
    if(S.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERrank") {
    ret = std::make_unique<TSample_impRank>(
      D_.generators, R, S_.bSampleSequences);
    if(S.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERerr") {
    ret = std::make_unique<TSample_impErr>(
      D_.generators, R, S_.bSampleSequences);
    if(S.bSampleSequences) die("Change importance sampling algorithm");
  }

  if(S.dataSamplingAlgo == "PERseq") ret = std::make_unique<Sample_impSeq>(
    D_.generators, R, S_.bSampleSequences);

  assert(ret not_eq nullptr);
  return std::move(ret);
}

}
