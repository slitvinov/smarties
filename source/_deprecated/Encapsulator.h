//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#ifndef smarties_Encapsulator_h
#define smarties_Encapsulator_h

#include "../ReplayMemory/MemoryBuffer.h"
#include "Network.h"

namespace smarties
{

class Builder;
class Optimizer;
class Network;

struct Encapsulator
{
  const std::string name;
  const Settings& settings;
  const Uint nThreads = settings.nThreads+settings.nAgents;
  const Uint nAppended = settings.appendedObs;
  const int ESpopSize = settings.ESpopSize;

  THRvec<std::vector<Activation*>> series =
                                THRvec<std::vector<Activation*>>(nThreads);
  THRvec<std::vector<Activation*>> series_tgt =
                                THRvec<std::vector<Activation*>>(nThreads);
  THRvec<int> first_sample = THRvec<int>(nThreads, -1);
  THRvec<int> error_placements = THRvec<int>(nThreads, -1);
  THRvec<Sequence*> thread_seq = THRvec<Sequence*>(nThreads, nullptr);

  // For CMAES based optimization. Keeps track of total loss associate with
  // Each weight vector sample:
  mutable Rvec losses = Rvec(ESpopSize, 0);

  mutable std::atomic<Uint> nAddedGradients{0};
  Uint nReducedGradients = 0;
  MemoryBuffer* const data;
  Optimizer* opt = nullptr;
  Network* net = nullptr;

  inline Uint nOutputs() const {
    if(net==nullptr) return data->sI.dimUsed*(1+nAppended);
    else return net->getnOutputs();
  }

  Encapsulator(const string N,const Settings&S,MemoryBuffer*const M);

  void initializeNetwork(Network* _net, Optimizer* _opt);

  void prepare(Sequence*const traj, const Uint len, const Uint samp, const Uint thrID);

  inline int mapTime2Ind(const Uint samp, const Uint thrID) const {
    assert(first_sample[thrID]<=(int)samp);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    const int ind = (int)samp - first_sample[thrID];
    return ind;
  }

  Rvec state2Inp(const int t, const Uint thrID) const;

  Rvec forward(const int samp, const Uint thrID, const int wghtID) const;

  void backward(const Rvec&error, const Uint samp, const Uint thrID) const;

  void prepareUpdate();

  void applyUpdate();

  void gradient(const Uint thrID) const;

  void save(const std::string base, const bool bBackup);
  void restart(const std::string base = std::string());

  void getHeaders(std::ostringstream& buff) const;
  void getMetrics(std::ostringstream& buff) const;
};

} // end namespace smarties
#endif // smarties_Quadratic_term_h



struct Encapsulator
{
  const bool bRecurrent = settings.bRecurrent, nMaxBPTT = settings.nnBPTTseq;
  const int ESpopSize = settings.ESpopSize;
  const Uint nAgents = settings.nAgents;
  const std::string name;
  const Settings& settings;
  const Uint nThreads = settings.nThreads+settings.nAgents;
  const Uint nAppended = settings.appendedObs;
  const int ESpopSize = settings.ESpopSize;

  THRvec<std::vector<Activation*>> series =
                                THRvec<std::vector<Activation*>>(nThreads);
  THRvec<std::vector<Activation*>> series_tgt =
                                THRvec<std::vector<Activation*>>(nThreads);
  THRvec<int> first_sample = THRvec<int>(nThreads, -1);
  THRvec<int> error_placements = THRvec<int>(nThreads, -1);
  THRvec<Sequence*> thread_seq = THRvec<Sequence*>(nThreads, nullptr);

  // For CMAES based optimization. Keeps track of total loss associate with
  // Each weight vector sample:
  mutable Rvec losses = Rvec(ESpopSize, 0);

  mutable std::atomic<Uint> nAddedGradients{0};
  Uint nReducedGradients = 0;
  MemoryBuffer* const data;

  inline Uint nOutputs() const {
    if(net==nullptr) return data->sI.dimUsed*(1+nAppended);
    else return net->getnOutputs();
  }

  Encapsulator(const string N,const Settings&S,MemoryBuffer*const M);

  void initializeNetwork(Network* _net, Optimizer* _opt);

  void prepare(Sequence*const traj, const Uint len, const Uint samp, const Uint thrID);

  inline int mapTime2Ind(const Uint samp, const Uint thrID) const {
    assert(first_sample[thrID]<=(int)samp);
    //ind is mapping from time stamp along trajectoy and along alloc memory
    const int ind = (int)samp - first_sample[thrID];
    return ind;
  }

  Rvec state2Inp(const int t, const Uint thrID) const;

  Rvec forward(const int samp, const Uint thrID, const int wghtID) const;

  void backward(const Rvec&error, const Uint samp, const Uint thrID) const;

  void prepareUpdate();

  void applyUpdate();

  void gradient(const Uint thrID) const;

  void save(const std::string base, const bool bBackup);
  void restart(const std::string base = std::string());

  void getHeaders(std::ostringstream& buff) const;
  void getMetrics(std::ostringstream& buff) const;
};


Rvec Approximator::forward(const Uint samp, const Uint thrID,
  const int USE_WGT, const int USE_ACT, const int overwrite) const
{
  if(USE_ACT>0) assert( (Uint) USE_ACT <= extraAlloc );
  // To handle Relay calling to answer agents' requests:
  if(thrID>=nThreads) return forward_agent(thrID-nThreads);

  const Uint netID = thrID + USE_ACT*nThreads;
  const std::vector<Activation*>& act = USE_ACT>=0? series[netID]
                                                  : series_tgt[thrID];
  const std::vector<Activation*>& act_cur = series[thrID];
  const int ind = mapTime2Ind(samp, thrID);

  //if already computed just give answer
  if(act[ind]->written && not overwrite) return act[ind]->getOutput();

  // write previous outputs if needed (note: will spawn nested function calls)
  // previous output use the same weights only if not target weights
  if(ind>0 && not act_cur[ind-1]->written)
    forward(samp-1, thrID, std::max(USE_WGT, 0), 0);

  const Rvec inp = getInput(samp, thrID, USE_WGT);
  //cout <<"USEW : "<< USE_WGT << endl; fflush(0);
  return getOutput(inp, ind, act[ind], thrID, USE_WGT);
}

Rvec Approximator::getInput(const Uint samp, const Uint thrID, const int USEW) const
{
  Rvec inp = input->forward(samp, thrID, USEW);
  if(relay not_eq nullptr) {
    const Rvec addedinp = relay->get(samp, thrID, USEW);
    assert(addedinp.size());
    inp.insert(inp.end(), addedinp.begin(), addedinp.end());
    //if(!thrID) cout << "relay "<<print(addedinp) << endl;
  }
  assert(inp.size() == net->getnInputs());
  return inp;
}

Rvec Approximator::getOutput(const Rvec inp, const int ind,
  Activation*const act, const Uint thrID, const int USEW) const
{
  //hardcoded to use time series predicted with cur weights for recurrencies:
  const std::vector<Activation*>& act_cur = series[thrID];
  const Activation*const recur = ind? act_cur[ind-1] : nullptr;
  assert(USEW < (int) net->sampled_weights.size() );
  const Parameters* const W = opt->getWeights(USEW);
  assert( W not_eq nullptr );
  const Rvec ret = net->predict(inp, recur, act, W);
  //if(!thrID) cout<<"net fwd with inp:"<<print(inp)<<" out:"<<print(ret)<<endl;
  act->written = true;
  return ret;
}



void Approximator::allocMorePerThread(const Uint nAlloc)
{
  assert(nAlloc > 0 && extraAlloc == 0);
  extraAlloc = nAlloc;
  assert(opt not_eq nullptr && net not_eq nullptr);
  series.resize(nThreads*(1+nAlloc));

  for (Uint j=1; j<=nAlloc; j++)
    #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
      for (Uint i = j*nThreads; i<(1+j)*nThreads; i++)
        #pragma omp critical
          series[i].reserve(MAX_SEQ_LEN);
}

void Approximator::prepare_seq(Sequence*const traj, const Uint thrID,
  const Uint wghtID) const
{
  if(error_placements[thrID] > 0) die("");
  input->prepare(traj, traj->nsteps(), 0, thrID);

  for(Uint k=0; k < 1+extraAlloc; k++)
    net->prepForBackProp(series[thrID + k*nThreads], traj->nsteps());

  if(series_tgt.size()>thrID)
    net->prepForFwdProp(series_tgt[thrID], traj->nsteps());

  first_sample[thrID] = 0;
  thread_Wind[thrID] = wghtID;
  thread_seq[thrID] = traj;
}

void Approximator::prepare_one(Sequence*const traj, const Uint samp,
    const Uint thrID, const Uint wghtID) const
{
  if(error_placements[thrID] > 0) die("");
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const Uint nTotal = nRecurr + 2;

  input->prepare(traj, nTotal, samp - nRecurr, thrID);

  for(Uint k=0; k < 1+extraAlloc; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  net->prepForFwdProp(series_tgt[thrID], nTotal);

  first_sample[thrID] = samp - nRecurr;
  thread_Wind[thrID] = wghtID;
  thread_seq[thrID] = traj;
}

void Approximator::prepare(Sequence*const traj, const Uint samp,
    const Uint N, const Uint thrID, const Uint wghtID) const
{
  if(error_placements[thrID] > 0) die("");
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const Uint nTotal = nRecurr + 1 + N;

  input->prepare(traj, nTotal, samp - nRecurr, thrID);

  for(Uint k=0; k < 1+extraAlloc; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  net->prepForFwdProp(series_tgt[thrID], nTotal);

  first_sample[thrID] = samp - nRecurr;
  thread_Wind[thrID] = wghtID;
  thread_seq[thrID] = traj;
}

void Approximator::prepare(const MiniBatch& B,
                           const Agent& agent,
                           const Sint wghtID) const
{
  assert(agentsContexts.size() > agent.ID);
  AgentContext & C = agentsContexts[agent.ID];
  C.load(net, B, agent, wghtID);
  //learner->select always only gets one new state, so we assume that it needs
  //to run one (or more) forward net at time t, so here also compute recurrency
  if(preprocessing) preprocessing->prepare(B, agent, wghtID);
  if(auxInputNet) auxInputNet->prepare(B, agent, wghtID);
  // if using relays, ask for previous actions, to be used for recurrencies
  // why? because the past is the past.
  const Parameters* const W = opt->getWeights(wghtID);
  //Advance recurr net with 0 initialized activations for nRecurr steps
  for(Uint i=0, t=stepid-nRecurr; i<nRecurr; i++, t++)
    net->predict(getInput(t,fakeThrID,wghtID), i? act[i-1]:nullptr, act[i], W);
}

Rvec Approximator::forward(const Agent& agent,
                           const Uint t,
                                 Sint sampID) const
{
  // assume we already computed recurrencies
  const std::vector<Activation*>& act = agent_series[agentID];
  const int fakeThrID = nThreads + agentID, wghtID = agent_Wind[agentID];
  const Uint stepid = agent_seq[agentID]->ndata();
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, stepid) : 0;
  if(act[nRecurr]->written) return act[nRecurr]->getOutput();
  const Parameters* const W = opt->getWeights(wghtID);
  const Rvec inp = getInput(stepid, fakeThrID, wghtID);
  return net->predict(inp, nRecurr? act[nRecurr-1] : nullptr, act[nRecurr], W);
}
