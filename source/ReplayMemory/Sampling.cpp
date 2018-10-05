//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#include "MemoryBuffer.h"
#include "Sampling.h"

Sampling::Sampling(const Settings& S, const MemoryBuffer*const R) :
gens(S.generators), RM(R), Set(RM->Set) {}

long Sampling::nSequences() const { return RM->readNSeq(); }
long Sampling::nTransitions() const { return RM->readNData(); }

TSample_uniform::TSample_uniform(const Settings&S, const MemoryBuffer*const R):
  Sampling(S,R) {}
TSample_impLen::TSample_impLen(const Settings&S, const MemoryBuffer*const R):
  Sampling(S,R) {}
SSample_uniform::SSample_uniform(const Settings&S, const MemoryBuffer*const R):
  Sampling(S,R) {}

void TSample_uniform::sample(vector<Uint>& seq, vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");

  // Drawing of samples is either uniform (each sample has same prob)
  // or based on importance sampling. The latter is TODO
  const long nSeqs = nSequences();
  const long nData = nTransitions();
  std::uniform_int_distribution<Uint> distObs(0, nData-1);

  std::vector<Uint> ret(seq.size());
  std::vector<Uint>::iterator it = ret.begin();
  while(it not_eq ret.end())
  {
    std::generate(it, ret.end(), [&]() { return distObs(gens[0]); } );
    std::sort(ret.begin(), ret.end());
    it = std::unique (ret.begin(), ret.end());
  } // ret is now also sorted!

  // go through each element of ret to find corresponding seq and obs
  for (Uint k = 0, cntO = 0, i = 0; k<nSeqs; k++) {
    while(1) {
      assert(ret[i] >= cntO && i < seq.size());
      if(ret[i] < cntO + Set[k]->ndata()) { // is ret[i] in sequence k?
        obs[i] = ret[i] - cntO; //if ret[i]==cntO then obs 0 of k and so forth
        seq[i] = k;
        i++; // next iteration remember first i-1 were already found
      }
      else break;
      if(i == seq.size()) break; // then found all elements of sequence k
    }
    //assert(cntO == Set[k]->prefix);
    if(i == seq.size()) break; // then found all elements of ret
    cntO += Set[k]->ndata(); // advance observation counter
    if(k+1 == nSeqs) die(" "); // at last iter we must have found all
  }
}
void TSample_uniform::prepare() {}

void SSample_uniform::sample(vector<Uint>& seq, vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");
  const long nSeqs = nSequences(), nBatch = obs.size();;
  std::fill(seq.begin(), seq.end(), 0);
  std::uniform_int_distribution<Uint> distSeq(0, nSeqs-1);
  std::vector<Uint>::iterator it = seq.begin();
  while(it not_eq seq.end())
  {
    std::generate(it, seq.end(), [&]() { return distSeq(gens[0]); } );
    std::sort( seq.begin(), seq.end() );
    it = std::unique( seq.begin(), seq.end() );
  }
  const auto compare = [&](const Uint a, const Uint b) {
    return Set[a]->ndata() > Set[b]->ndata();
  };
  std::sort(seq.begin(), seq.end(), compare);
  for (Uint i=0; i<nBatch; i++) obs[i] = Set[seq[i]]->ndata() - 1;
}
void SSample_uniform::prepare() {}

void TSample_impLen::sample(vector<Uint>& seq, vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");

  const Uint nBatch = obs.size();
  std::uniform_real_distribution<float> distT(0, 1);
  std::vector<std::pair<Uint, Uint>> S (nBatch);

  std::vector<std::pair<Uint, Uint>>::iterator it = S.begin();
  while(it not_eq S.end()) {
    std::generate(it, S.end(), [&] () {
        const Uint _s = dist(gens[0]), _t = distT(gens[0]) * Set[_s]->ndata();
        return std::pair<Uint, Uint> {_s, _t};
      }
    );
    std::sort( S.begin(), S.end() );
    it = std::unique( S.begin(), S.end() );
  }

  for (Uint i=0; i<nBatch; i++) { seq[i] = S[i].first; obs[i] = S[i].second; }
}
void TSample_impLen::prepare()
{
  const Uint nSeqs = nSequences();
  std::vector<float> probs(nSeqs, 0);
  #pragma omp parallel for schedule(static)
  for (Uint i=0; i<nSeqs; i++) probs[i] = Set[i]->ndata();
  dist = std::discrete_distribution<Uint>(probs.begin(), probs.end());
}

/*
class TSample_Tpriority : public Sampling
{
  TSample_uniform() : Sampling() {}
  void sample(vector<Uint>& seq, vector<Uint>& obs) override
  {
    if(seq.size() not_eq obs.size()) die(" ");

    // Drawing of samples is either uniform (each sample has same prob)
    // or based on importance sampling. The latter is TODO
    discrete_distribution<Uint> & distObs = distPER;

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
          obs[i] = ret[i] - cntO; //if ret[i]==cntO then obs 0 of k and so forth
          seq[i] = k;
          i++; // next iteration remember first i-1 were already found
        }
        else break;
        if(i == seq.size()) break; // then found all elements of sequence k
      }
      //assert(cntO == Set[k]->prefix);
      if(i == seq.size()) break; // then found all elements of ret
      cntO += Set[k]->ndata(); // advance observation counter
      if(k+1 == Set.size()) die(" "); // at last iter we must have found all
    }
  }
};
*/

/*
void MemoryBuffer::sampleTransitions2(vector<Uint>& seq, vector<Uint>& obs)
{
  if(seq.size() not_eq obs.size()) die(" ");

  // Drawing of samples is either uniform (each sample has same prob)
  // or based on importance sampling. The latter is TODO
  const long nSeq = nSequences;
  std::uniform_int_distribution<Uint> distObs(0, nSeq-1);

  std::vector<Uint> ret(seq.size());
  std::vector<Uint>::iterator it = ret.begin();
  while(it not_eq ret.end())
  {
    std::generate(it, ret.end(), [&]() { return distObs(generators[0]); } );
    std::sort(ret.begin(), ret.end());
    it = std::unique (ret.begin(), ret.end());
  } // ret is now also sorted!

  // go through each element of ret to find corresponding seq and obs
  for (Uint k = 0, cntO = 0, i = 0; k<Set.size(); k++)
  {
    while(1) {
      assert(ret[i] >= cntO && i < seq.size());
      if(ret[i] < cntO + Set[k]->ndata()) { // is ret[i] in sequence k?
        obs[i] = ret[i] - cntO; //if ret[i]==cntO then obs 0 of k and so forth
        seq[i] = k;
        i++; // next iteration remember first i-1 were already found
      }
      else break;
      if(i == seq.size()) break; // then found all elements of sequence k
    }
    //assert(cntO == Set[k]->prefix);
    if(i == seq.size()) break; // then found all elements of ret
    cntO += Set[k]->ndata(); // advance observation counter
    if(k+1 == Set.size()) die(" "); // at last iter we must have found all
  }
}
*/


#if 0
void MemoryBuffer::fillSamples()
{
  #ifdef ALT_SAMPLING
  const long nSeq = nSequences;
  const long nObs = nTransitions;
  samples.resize(nObs);

  for(long i=0, locPrefix=0; i<nSeq; i++) {
    Set[i]->prefix = locPrefix;
    locPrefix += Set[i]->ndata();
  }
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < nSeq; i++)
    for(Uint j=0, k=Set[i]->prefix; j<Set[i]->ndata(); ++j, ++k)
      samples[k] = std::pair<unsigned, unsigned>{i, j};

  #if 0
  std::vector<std::mt19937>& gens = settings.generators;
  __gnu_parallel::random_shuffle(samples.begin(), samples.end(), gens[0]);
  #else
  const Uint nThreads = settings.nThreads;
  std::vector<Uint> thdStarts(nThreads, 0);
  const long stride = std::ceil(nObs / (Real) nThreads);
  std::vector<std::mt19937>& gens = settings.generators;
  #pragma omp parallel num_threads(nThreads)
  {
    const int thrI = omp_get_thread_num();
    {
      const long start = thrI*stride;
      const long end = std::min( (thrI+1)*stride, nObs);
      std::shuffle(samples.begin()+start, samples.begin()+end, gens[thrI]);
    }
    #pragma omp barrier
    // now each thread gets a quantile of partial sorts
    std::vector<std::pair<unsigned, unsigned>> tmp;
    tmp.reserve(stride); // because we want to push back
    for(Uint t=0; t<nThreads; t++)
    {
      const Uint i = (t + thrI) % nThreads;
      const Uint start = i*stride, end = std::min( (i+1)*stride, nObs);
      #pragma omp for schedule(static) nowait // equally divided
      for(Uint j=start; j<end; j++) tmp.push_back( samples[j] );
    }
    const Uint locSize = tmp.size();
    thdStarts[thrI] = locSize;
    std::shuffle(tmp.begin(), tmp.end(), gens[thrI]);

    #pragma omp barrier // wait all those thdStarts values

    Uint threadStart = 0;
    for(int i=0; i<thrI; i++) threadStart += thdStarts[i];
    for(Uint i=0; i<locSize; i++) samples[i+threadStart] = tmp[i];
  }
  #endif
  #endif
}
  #if 0
    for(Uint i=0; i<seq.size(); ++i)
    {
      seq[i] = samples.back().first;
      obs[i] = samples.back().second;
      samples.pop_back();
    }
  #endif
#endif

  #ifdef PRIORITIZED_ER
    vector<float> probs(nTransitions.load(), 1);
    distPER = discrete_distribution<Uint>(probs.begin(), probs.end());
  #endif
