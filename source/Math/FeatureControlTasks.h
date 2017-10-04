/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Quadratic_advantage.h"
#include "../Learners/Learner.h"
/*
class ContinuousFeatureControl
{
protected:
  //these are all feature controls that use NAF stile continuous q learning
  //they train the network to output V, matrix and mean.
  //We just need to know which outputs if the network belong to this task:
  //therefore determined first index, nA and nL
  const Uint outIndex, nA, nL;
  const Network*const net;
  const Transitions*const data;
  static inline Uint compute_nL(const Uint NA)
  {
    return (NA*NA + NA)/2;
  }
  inline bool bTerminal(const Uint seq, const Uint samp) const
  {
    return samp+2==data->Set[seq]->tuples.size() && data->Set[seq]->ended;
  }

public:
  ContinuousFeatureControl(Uint oi, Uint na, const Network*const _net,
    const Transitions*const d) : outIndex(oi), nA(na), nL(compute_nL(na)),
    net(_net), data(d)  { }
};

class ContinuousSignControl: public ContinuousFeatureControl
{
  const Uint nS;
  const vector<Uint> net_outputs = {nS, nS*nL, nS*nA};
  const vector<Uint> net_indices = {outIndex, outIndex+nS, outIndex+nS*(1+nL)};
  const Uint nOutputs = nS*(1+nL+nA);

  inline Quadratic_advantage prepare_advantage(const vector<Real>& out,
    const Uint iS) const
  {
    const Uint startL=net_indices[1] + iS*nL, startA=net_indices[2] + iS*nA;
    return Quadratic_advantage(startL, startA, nA, nL, out);
  }
  inline Real computeReward(const vector<Real>sold, const vector<Real> scur,
    const vector<Real> snew, const Uint iS) const
  {
    if((scur[iS]-sold[iS])*(snew[iS]-scur[iS])<=0)
      return fabs(snew[iS]-scur[iS]);
    else return 0;
  }

public:
  ContinuousSignControl(Uint oi, Uint na, Uint ns, const Network*const _net,
    const Transitions*const d) : ContinuousFeatureControl(oi,na,_net,d), nS(ns)
    {
      printf("ContinuousSignControl: train network outputs: %s %s\n",
        print(net_indices).c_str(),print(net_outputs).c_str());
    }

  inline void Train(
    const Activation*const nPrev, const Activation*const nNext,
    const vector<Real>&act, const Uint seq, const Uint samp,
    const Real gamma, vector<Real>& grad) const
  {
    if(samp == 0) {
      for (Uint j=net_indices[0]; j<net_indices[0]+nOutputs; j++) grad[j] = 0;
      return;
    }
    const Tuple* const t0 = data->Set[seq]->tuples[samp-1];
    const Tuple* const t1 = data->Set[seq]->tuples[samp];
    const Tuple* const t2 = data->Set[seq]->tuples[samp+1];
    const vector<Real> sold = data->standardize(t0->s);
    const vector<Real> scur = data->standardize(t1->s);
    const vector<Real> snew = data->standardize(t2->s);
    const bool bTerm = bTerminal(seq,samp);
    const vector<Real> outPrev = net->getOutputs(nPrev);
    const vector<Real> outNext =bTerm? vector<Real>() : net->getOutputs(nNext);
    for (Uint j=0; j<nS; j++) {
      const Real rew = computeReward(sold, scur, snew, j);
      const Quadratic_advantage adv = prepare_advantage(outPrev, j);
      const Real Qsold = outPrev[net_indices[0]+j] + adv.computeAdvantage(act);
      const Real value =(bTerm)? rew : rew + gamma*outNext[net_indices[0]+j];
      grad[net_indices[0]+j] = value-Qsold;
      adv.grad(act, value-Qsold, grad, data->aI.bounded);
    }
  }

  static Uint addRequestedLayers(const Uint NA, const Uint NS, vector<Uint>&net_indices, vector<Uint>&net_outputs, vector<Real>&out_weight_inits)
  {
    net_indices.push_back(net_indices.back()+net_outputs.back());
    net_outputs.push_back(NS);
    out_weight_inits.push_back(-1);
    const Uint task_out0 = net_indices.back();

    net_indices.push_back(net_indices.back()+net_outputs.back());
    net_outputs.push_back(NS*compute_nL(NA));
    out_weight_inits.push_back(-1);

    net_indices.push_back(net_indices.back()+net_outputs.back());
    net_outputs.push_back(NS*NA);
    out_weight_inits.push_back(-1);

    return task_out0;
  }
  static Uint addRequestedOutputs(const Uint NA, const Uint NS)
  {
    return NS*(1 + compute_nL(NA)+ NA);
  }
};

class DiscreteFeatureControl
{
protected:
  const Uint outIndex, nA;
  const Network*const net;
  const Transitions*const data;
  inline bool bTerminal(const Uint seq, const Uint samp) const
  {
    return samp+2==data->Set[seq]->tuples.size() && data->Set[seq]->ended;
  }

public:
  DiscreteFeatureControl(Uint oi, Uint na, const Network*const _net,
    const Transitions*const d) : outIndex(oi), nA(na), net(_net), data(d)  { }
};

class DiscreteSignControl: public DiscreteFeatureControl
{
  const Uint nS;
  const vector<Uint> net_outputs = {nS, nS*nA};
  const vector<Uint> net_indices = {outIndex, outIndex+nS};
  const Uint nOutputs = nS*(1+nA);
  inline Real computeReward(const vector<Real>sold, const vector<Real> scur,
    const vector<Real> snew, const Uint iS) const
  {
    if((scur[iS]-sold[iS])*(snew[iS]-scur[iS])<=0)
      return fabs(snew[iS]-scur[iS]);
    else return 0;
  }

public:
  DiscreteSignControl(Uint oi, Uint na, Uint ns, const Network*const _net,
    const Transitions*const d) : DiscreteFeatureControl(oi,na,_net,d), nS(ns)
    {
      printf("DiscreteSignControl: train network outputs: %s %s\n",
        print(net_indices).c_str(),print(net_outputs).c_str());
    }

  inline void Train(
    const Activation*const nPrev, const Activation*const nNext,
    const Uint act, const Uint seq, const Uint samp,
    const Real gamma, vector<Real>& grad) const
  {
    for (Uint j=net_indices[0]; j<net_indices[0]+nOutputs; j++) grad[j] = 0;
    if(samp == 0) return;
    assert(act<nA);
    const Tuple* const t0 = data->Set[seq]->tuples[samp-1];
    const Tuple* const t1 = data->Set[seq]->tuples[samp];
    const Tuple* const t2 = data->Set[seq]->tuples[samp+1];
    const vector<Real> sold = data->standardize(t0->s);
    const vector<Real> scur = data->standardize(t1->s);
    const vector<Real> snew = data->standardize(t2->s);
    const bool bTerm = bTerminal(seq,samp);
    const vector<Real> outPrev = net->getOutputs(nPrev);
    const vector<Real> outNext =bTerm? vector<Real>() : net->getOutputs(nNext);

    for (Uint j=0, k=net_indices[1]; j<nS; j++, k+=nA) {
      const Real rew = computeReward(sold, scur, snew, j);
      const Real Qsold = outPrev[net_indices[0]+j] + outPrev[k+act];
      const Real value =(bTerm)? rew : rew + gamma*outNext[net_indices[0]+j];
      grad[net_indices[0]+j] = grad[k+act] = value-Qsold;
    }
  }

  static Uint addRequestedLayers(const Uint NA, const Uint NS, vector<Uint>&net_indices, vector<Uint>&net_outputs, vector<Real>&out_weight_inits)
  {
    net_indices.push_back(net_indices.back()+net_outputs.back());
    net_outputs.push_back(NS);
    out_weight_inits.push_back(-1);
    const Uint task_out0 = net_indices.back();

    net_indices.push_back(net_indices.back()+net_outputs.back());
    net_outputs.push_back(NS*NA);
    out_weight_inits.push_back(-1);

    return task_out0;
  }
  static Uint addRequestedOutputs(const Uint NA, const Uint NS)
  {
    return NS*(1+NA);
  }
};
*/
