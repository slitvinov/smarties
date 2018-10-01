//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "MemoryBuffer.h"
#include <iterator>

MemoryBuffer::MemoryBuffer(const Settings&S, const Environment*const E):
 settings(S), env(E) {
  Set.reserve(settings.maxTotObsNum);
}

void MemoryBuffer::save(const string base, const Uint nStep, const bool bBackup)
{
  FILE * wFile = fopen((base+"scaling.raw").c_str(), "wb");
  fwrite(   mean.data(), sizeof(memReal),   mean.size(), wFile);
  fwrite( invstd.data(), sizeof(memReal), invstd.size(), wFile);
  fwrite(    std.data(), sizeof(memReal),    std.size(), wFile);
  fwrite(&invstd_reward, sizeof(Real),             1, wFile);
  fflush(wFile); fclose(wFile);

  if(bBackup) {
    ostringstream S; S<<std::setw(9)<<std::setfill('0')<<nStep;
    wFile = fopen((base+"scaling_"+S.str()+".raw").c_str(), "wb");
    fwrite(   mean.data(), sizeof(memReal),   mean.size(), wFile);
    fwrite( invstd.data(), sizeof(memReal), invstd.size(), wFile);
    fwrite(    std.data(), sizeof(memReal),    std.size(), wFile);
    fwrite(&invstd_reward, sizeof(Real),             1, wFile);
    fflush(wFile); fclose(wFile);
  }
}

void MemoryBuffer::restart(const string base)
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

    size_t size1 = fread(   mean.data(), sizeof(memReal),   mean.size(), wFile);
    size_t size2 = fread( invstd.data(), sizeof(memReal), invstd.size(), wFile);
    size_t size3 = fread(    std.data(), sizeof(memReal),    std.size(), wFile);
    size_t size4 = fread(&invstd_reward, sizeof(Real),             1, wFile);
    fclose(wFile);
    if(size1!=mean.size()|| size2!=invstd.size()|| size3!=std.size()|| size4!=1)
      _die("Mismatch in restarted file %s.", (base+"_scaling.raw").c_str());
  }
}

void MemoryBuffer::clearAll()
{
  std::lock_guard<std::mutex> lock(dataset_mutex);
  //delete already-used trajectories
  for(auto& old_traj: Set) _dispose_object(old_traj);

  Set.clear(); //clear trajectories used for learning
  nSequences = 0;
  nTransitions = 0;
}

void MemoryBuffer::sampleTransitions(vector<Uint>& seq, vector<Uint>& obs) {
  if(seq.size() not_eq obs.size()) die(" ");

  // Drawing of samples is either uniform (each sample has same prob)
  // or based on importance sampling. The latter is TODO
  #ifndef PRIORITIZED_ER
    std::uniform_int_distribution<Uint> distObs(0, readNData()-1);
  #else
    discrete_distribution<Uint> & distObs = distPER;
  #endif

  std::vector<Uint> ret(seq.size());
  std::vector<Uint>::iterator it = ret.begin();
  while(it not_eq ret.end())
  {
    std::generate(it, ret.end(), [&]() { return distObs(generators[0]); } );
    std::sort(ret.begin(), ret.end());
    it = std::unique (ret.begin(), ret.end());
  } // ret is now also sorted!

  // go through each element of ret to find corresponding seq and obs
  for (Uint k = 0, cntO = 0, i = 0; k<Set.size(); k++) {
    while(1) {
      assert(ret[i] >= cntO && i < seq.size());
      if(ret[i] < cntO + Set[k]->ndata()) { // is ret[i] in sequence k?
        obs[i] = ret[i] - cntO; // if ret[i]==cntO then obs 0 of k and so forth
        seq[i] = k;
        i++; // next iteration remember first i-1 were already found
      }
      else break;
      if(i == seq.size()) break; // then found all elements of sequence k
    }
    assert(cntO == Set[k]->prefix);
    if(i == seq.size()) break; // then found all elements of ret
    cntO += Set[k]->ndata(); // advance observation counter
    if(k+1 == Set.size()) die(" "); // at last iter we must have found all
  }
}

void MemoryBuffer::sampleSequences(vector<Uint>& seq) {
  std::fill(seq.begin(), seq.end(), 0);
  std::uniform_int_distribution<Uint> distSeq(0, readNSeq()-1);
  std::vector<Uint>::iterator it = seq.begin();
  while(it not_eq seq.end())
  {
    std::generate(it, seq.end(), [&]() { return distSeq(generators[0]); } );
    std::sort( seq.begin(), seq.end() );
    it = std::unique( seq.begin(), seq.end() );
  }
  const auto compare = [&](const Uint a, const Uint b) {
    return Set[a]->ndata() > Set[b]->ndata();
  };
  std::sort(seq.begin(), seq.end(), compare);
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
  _dispose_object(Set.back());
  Set.pop_back();
  assert(nSequences == Set.size());
}
void MemoryBuffer::pushBackSequence(Sequence*const seq)
{
  lock_guard<mutex> lock(dataset_mutex);
  assert( readNSeq() == Set.size() and seq not_eq nullptr);
  const auto ind = Set.size();
  const Uint prefix =
  Set.push_back(seq);
  Set[ind]->prefix = ind>0? Set[ind-1]->prefix +Set[ind-1]->ndata() : 0;
  nTransitions += seq->ndata();
  needs_pass = true;
  nSequences++;
  assert( readNSeq() == Set.size());
  //cout << "push back " << prefix << " " << Set[ind]->ndata() << endl;
}

void MemoryBuffer::initialize()
{
  // All sequences obtained before this point should share the same time stamp
  for(Uint i=0;i<Set.size();i++) Set[i]->ID = readNSeenSeq()-1;
  #ifdef PRIORITIZED_ER
    vector<float> probs(nTransitions.load(), 1);
    distPER = discrete_distribution<Uint>(probs.begin(), probs.end());
  #endif
}

MemoryBuffer::~MemoryBuffer()
{
  for (auto & trash : Set) _dispose_object( trash);
}

void MemoryBuffer::checkNData() {
  #ifndef NDEBUG
    Uint cntSamp = 0;
    for(Uint i=0; i<Set.size(); i++) {
      assert(Set[i] not_eq nullptr);
      cntSamp += Set[i]->ndata();
    }
    assert(cntSamp==nTransitions);
    assert(Set.size()==nSequences);
  #endif
}

#ifdef PRIORITIZED_ER
#include "MemoryBuffer_prioritizedER.cpp"
#endif
