/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Encapsulator.h"
#include "StatsTracker.h"

#include <list>

struct Aggregator;
class Builder;

enum PARAMS { CUR, TGT }; /* use CUR or TGT weights */
struct Approximator
{
  Settings& settings;
  const string name;
  const Uint nThreads, mpisize, nMaxBPTT = MAX_UNROLL_BFORE;
  const bool bRecurrent;

  Encapsulator* const input;
  MemoryBuffer* const data;
  const Aggregator* const relay;
  Optimizer* opt = nullptr;
  Network* net = nullptr;
  StatsTracker* gradStats = nullptr;
  vector<Parameters*> relayG;

  mutable vector<int> error_placements, first_sample;
  mutable Uint nAddedGradients=0, reducedGradients=0;
  Uint extraAlloc = 0;
  int relayInp = -1;
  //thread safe memory for prediction with current weights:
  mutable vector<vector<Activation*>> series;
  //thread safe  memory for prediction with target weights. Rules are that
  // index along the two alloc vectors is the same for the same sample, and
  // that tgt net (if available) takes recurrent activation from current net:
  mutable vector<vector<Activation*>> series_tgt;

  Approximator(const string _name, Settings& S, Encapsulator*const enc,
    MemoryBuffer* const data_ptr, const Aggregator* const r = nullptr) :
  settings(S), name(_name), nThreads(S.nThreads), mpisize(S.learner_size),
  bRecurrent(S.bRecurrent), input(enc), data(data_ptr), relay(r),
  relayG(nThreads, nullptr), error_placements(nThreads, -1),
  first_sample(nThreads, -1), series(nThreads), series_tgt(nThreads) {}

  Builder buildFromSettings(Settings& _s, const vector<Uint> n_outputs);
  Builder buildFromSettings(Settings& _s, const Uint n_outputs);

  void initializeNetwork(Builder& build, Real cutGradFactor = 5);
  void allocMorePerThread(const Uint nAlloc);

  void prepare_opc(const Sequence*const traj, const Uint samp,
      const Uint thrID, const Uint nSamples = 1) const;

  void prepare_seq(const Sequence*const traj, const Uint thrID,
    const Uint nSamples = 1) const;

  void prepare_one(const Sequence*const traj, const Uint samp,
      const Uint thrID, const Uint nSamples = 1) const;

  Rvec forward(const Sequence* const traj, const Uint samp,
    const Uint thrID, const PARAMS USE_WEIGHTS, const PARAMS USE_ACT,
    const Uint iSample, const int overwrite) const;

  // this is templated only to increase clarity when calling the forward op
  template <PARAMS USE_WEIGHTS=CUR, PARAMS USE_ACT=USE_WEIGHTS, int overwrite=0>
  inline Rvec forward(const Sequence* const traj, const Uint samp,
      const Uint thrID, const Uint iSample = 0) const
  {
    return forward(traj, samp, thrID, USE_WEIGHTS, USE_ACT, iSample, overwrite);
  }


  Rvec relay_backprop(const Rvec error, const Uint samp,
    const Uint thrID, const PARAMS USEW) const;

  template <PARAMS USEW = CUR>
  inline Rvec relay_backprop(const Rvec error, const Uint samp,
      const Uint thrID) const
  {
    return relay_backprop(error, samp, thrID, USEW);
  }

  Rvec forward_agent(const Sequence* const traj, const Agent& agent,
    const Uint thrID, const PARAMS USEW) const;

  template <PARAMS USEW = CUR>
  inline Rvec forward_agent(const Sequence* const traj,
    const Agent& agent, const Uint thrID) const
  {
    return forward_agent(traj, agent, thrID, USEW);
  }

  Rvec getOutput(const Rvec inp, const int ind,
    Activation*const act, const Uint thrID, const PARAMS USEW) const;

  template <PARAMS USEW = CUR>
  inline Rvec getOutput(const Rvec inp, const int ind, Activation*const act, const Uint thrID) const
  {
    return getOutput(inp, ind, act, thrID, USEW);
  }

  Rvec getInput(const Sequence*const traj, const Uint samp,
    const Uint thrID) const;

  inline int mapTime2Ind(const Uint samp, const Uint thrID) const
  {
    assert(first_sample[thrID]<=(int)samp);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    const int ind = (int)samp - first_sample[thrID];
    return ind;
  }

  template <PARAMS USEW = CUR>
  inline Rvec get(const Sequence*const traj, const Uint samp,
    const Uint thrID)
  {
    const vector<Activation*>&act =USEW==CUR? series[thrID] : series_tgt[thrID];
    return act[mapTime2Ind(samp, thrID)]->getOutput();
  }

  void backward(Rvec error, const Uint samp, const Uint thrID,
    const Uint iSample = 0) const;

  void prepareUpdate();
  void applyUpdate();

  void gradient(const Uint thrID) const;

  inline Uint nOutputs() const
  {
   return net->getnOutputs();
  }

  void getMetrics(ostringstream& buff) const;
  void getHeaders(ostringstream& buff) const;

  void save(const string base = string())
  {
    if(opt == nullptr) die("Attempted to save uninitialized net!");
    opt->save(base + name);
  }
  void restart(const string base = string())
  {
    if(opt == nullptr) die("Attempted to restart uninitialized net!");
    opt->restart(base+name);
  }

  inline void updateGradStats(const Uint iter) const
  {
    gradStats->reduce_stats(iter);
  }
};

enum RELAY { VEC, ACT, NET};
//TODO stepid
struct Aggregator
{
  const bool bRecurrent;
  const Uint nThreads, nOuts, nMaxBPTT = MAX_UNROLL_BFORE;
  const MemoryBuffer* const data;
  const ActionInfo& aI = data->aI;
  const Approximator* const approx;

  mutable vector<int> first_sample;
  mutable vector<vector<Rvec>> inputs; // [thread][time][component]
  mutable vector<RELAY> usage; // [thread]

  // Settings file, the memory buffer class from which all trajectory pointers
  // will be drawn from, the number of outputs from the aggregator. If 0 then
  // 1) output the actions of the sequence (default)
  // 2) output the result of NN approximator (pointer a)
  Aggregator(Settings& S, const MemoryBuffer*const d, const Uint nOut=0,
    const Approximator*const a = nullptr): bRecurrent(S.bRecurrent),
    nThreads(S.nThreads), nOuts(nOut? nOut: d->aI.dim), data(d), approx(a),
    first_sample(nThreads, -1), inputs(nThreads), usage(nThreads, ACT) { }

  void prepare(const RELAY SET, const Uint thrID) const;

  void prepare_opc(const Sequence*const traj, const Uint samp,
      const Uint thrID, const RELAY SET = VEC) const;

  void prepare_seq(const Sequence*const traj, const Uint thrID,
    const RELAY SET = VEC) const;

  void prepare_one(const Sequence*const traj, const Uint samp,
      const Uint thrID, const RELAY SET = VEC) const;

  void set(const Rvec vec,const Uint samp,const Uint thrID) const;

  Rvec get(const Sequence*const traj, const Uint samp,
      const Uint thrID) const;

  inline Uint nOutputs() const
  {
    #ifndef NDEBUG
      if(usage[omp_get_thread_num()] == VEC) {
        assert(inputs[omp_get_thread_num()][0].size() == nOuts);
      } else if (usage[omp_get_thread_num()] == NET) {
        assert(approx not_eq nullptr);
        assert(approx->nOutputs() == nOuts);
      } else assert(aI.dim == nOuts);
    #endif
    return nOuts;
  }
};
