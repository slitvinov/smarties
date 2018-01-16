/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Approximator.h"
#include "../Network/Builder.h"

void Aggregator::prepare(const RELAY SET, const Uint thrID) const
{
  usage[thrID] = SET;
}

void Aggregator::prepare_opc(const Sequence*const traj, const Uint samp,
    const Uint thrID, const RELAY SET) const
{
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // predict pol/val of all states from samp to T (T-1 if T is term state)
  const Uint nSValues =  traj->tuples.size() - samp - traj->ended;
  const Uint nTotal = nRecurr + nSValues;
  first_sample[thrID] = samp - nRecurr;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, vector<Real>());
  usage[thrID] = SET;
}

void Aggregator::prepare_seq(const Sequence*const traj, const Uint thrID, const RELAY SET) const
{
  const Uint nSValues =  traj->tuples.size() - traj->ended;
  first_sample[thrID] = 0;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nSValues, vector<Real>());
  usage[thrID] = SET;
}

void Aggregator::prepare_one(const Sequence*const traj, const Uint samp,
    const Uint thrID, const RELAY SET) const
{
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  const Uint nSValues = terminal ? 1 : 2; //probably faster to just assume 2
  const Uint nTotal = nRecurr + nSValues;
  first_sample[thrID] = samp - nRecurr;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, vector<Real>());
  usage[thrID] = SET;
}

void Aggregator::set(const vector<Real> vec,const Uint samp,const Uint thrID) const
{
  usage[thrID] = VEC;
  const int ind = (int)samp - first_sample[thrID];
  assert(first_sample[thrID] <= (int)samp);
  assert(ind >= 0 && (int) inputs[thrID].size() > ind);
  inputs[thrID][ind] = vec;
}

vector<Real> Aggregator::get(const Sequence*const traj, const Uint samp,
    const Uint thrID) const
{
  if(usage[thrID] == VEC) {
    assert(first_sample[thrID] >= 0);
    const int ind = (int)samp - first_sample[thrID];
    assert(first_sample[thrID] <= (int)samp);
    assert(ind >= 0 && (int) inputs[thrID].size() > ind);
    assert(inputs[thrID][ind].size());
    return inputs[thrID][ind];
  } else if (usage[thrID] == ACT) {
    return aI.getInvScaled(traj->tuples[samp]->a);
  } else {
    return approx->forward<CUR>(traj, samp, thrID);
  }
}

Builder Approximator::buildFromSettings(Settings&sett, const vector<Uint>nouts)
{
  Builder build(sett);
  Uint nInputs = input->nOutputs() + (relay==nullptr ? 0 : relay->nOutputs());
  build.stackSimple( nInputs, nouts );
  return build;
}

Builder Approximator::buildFromSettings(Settings& _s, const Uint n_outputs) {
  Builder build(_s);
  Uint nInputs = input->nOutputs() + (relay==nullptr ? 0 : relay->nOutputs());
  build.stackSimple( nInputs, {n_outputs} );
  return build;
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
          series[i].reserve(settings.maxSeqLen);
}

void Approximator::initializeNetwork(Builder& build)
{
  net = build.build();
  opt = build.opt;
  assert(opt not_eq nullptr && net not_eq nullptr);

  #pragma omp parallel for schedule(static, 1) num_threads(nThreads)
  for (Uint i=0; i<nThreads; i++) // numa aware allocation
   #pragma omp critical
   {
     series[i].reserve(settings.maxSeqLen);
     series_tgt[i].reserve(settings.maxSeqLen);
     if(relay not_eq nullptr)
      extra_grads[i] = net->allocateParameters();
   }

  #ifdef __CHECK_DIFF //check gradients with finite differences
    net->checkGrads();
  #endif
  gradStats = new StatsTracker(net->getnOutputs(), name+"_grads", settings);
}

void Approximator::prepare_opc(const Sequence*const traj, const Uint samp,
    const Uint thrID, const Uint nSamples) const
{
  if(error_placements[thrID] > 0) gradient(thrID);
  assert(nSamples<=1+extraAlloc && nSamples>0);
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // predict pol/val of all states from samp to T (T-1 if T is term state)
  const Uint nSValues =  traj->tuples.size() - samp - traj->ended;
  const Uint nTotal = nRecurr + nSValues;
  input->prepare(nTotal, samp - nRecurr, thrID);

  for(Uint k=0; k<nSamples; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  net->prepForFwdProp(series_tgt[thrID], nTotal);

  error_placements[thrID] = -1;
  first_sample[thrID] = samp - nRecurr;
}

void Approximator::prepare_seq(const Sequence*const traj, const Uint thrID,
  const Uint nSamples) const
{
  if(error_placements[thrID] > 0) gradient(thrID);
  assert(nSamples<=1+extraAlloc && nSamples>0);
  const Uint nSValues =  traj->tuples.size() - traj->ended;
  input->prepare(nSValues, 0, thrID);

  for(Uint k=0; k<nSamples; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nSValues);

  if(series_tgt.size()>thrID)
    net->prepForFwdProp(series_tgt[thrID], nSValues);

  error_placements[thrID] = -1;
  first_sample[thrID] = 0;
}

void Approximator::prepare_one(const Sequence*const traj, const Uint samp,
    const Uint thrID, const Uint nSamples) const
{
  if(error_placements[thrID] > 0) gradient(thrID);
  assert(nSamples<=1+extraAlloc && nSamples>0);
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  const Uint nSValues = terminal ? 1 : 2; //probably faster to just assume 2
  const Uint nTotal = nRecurr + nSValues;

  input->prepare(nTotal, samp - nRecurr, thrID);
  for(Uint k=0; k<nSamples; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  net->prepForFwdProp(series_tgt[thrID], nTotal);

  error_placements[thrID] = -1;
  first_sample[thrID] = samp - nRecurr;
}

vector<Real> Approximator::forward(const Sequence* const traj, const Uint samp,
  const Uint thrID, const PARAMS USE_WEIGHTS, const PARAMS USE_ACT,
  const Uint iSample, const int overwrite) const
{
  if(iSample) assert(USE_ACT == CUR && iSample<=extraAlloc);
  const Uint netID = thrID + iSample*nThreads;
  const vector<Activation*>& act=USE_ACT==CUR? series[netID] :series_tgt[thrID];
  const vector<Activation*>& act_cur = series[thrID];
  const int ind = mapTime2Ind(samp, thrID);

  //if already computed just give answer
  if(act[ind]->written == true && not overwrite)
    return act[ind]->getOutput();

  // write previous outputs if needed (note: will spawn nested function calls)
  if(ind>0 && act_cur[ind-1]->written not_eq true)
    this->forward(traj, samp-1, thrID);

  const vector<Real> inp = getInput(traj, samp, thrID);
  return getOutput(inp, ind, act[ind], thrID, USE_WEIGHTS);
}

vector<Real> Approximator::relay_backprop(const vector<Real> error,
  const Uint samp, const Uint thrID, const PARAMS USEW) const
{
  if(relay == nullptr) die("Called relay_backprop without a relay.");
  const vector<Activation*>& act_tgt = series_tgt[thrID];
  const int ind = mapTime2Ind(samp, thrID), nInp = input->nOutputs();
  assert(act_tgt[ind]->written == true && relay not_eq nullptr);
  act_tgt[ind]->clearErrors();
  act_tgt[ind]->setOutputDelta(error);
  const Parameters*const W = USEW==CUR? net->weights : net->tgt_weights;
  net->backProp(nullptr, act_tgt[ind], nullptr, extra_grads[thrID], W);
  const vector<Real> gradR = act_tgt[ind]->getInputGradient();
  assert(gradR.size() == nInp + relay->nOutputs());
  return vector<Real>(&gradR[0]+nInp, &gradR[0]+nInp+relay->nOutputs());
}

vector<Real> Approximator::forward_agent(const Sequence* const traj,
  const Agent& agent, const Uint thrID, const PARAMS USEW) const
{
  if(error_placements[thrID] > 0) gradient(thrID);

  const Uint stepid = traj->ndata();
  input->prepare(1, stepid, thrID);
  net->prepForFwdProp(series[thrID], 2);

  const vector<Activation*>& act = series[thrID];
  const vector<Real> inp = getInput(traj, stepid, thrID);
  const Parameters* const W = USEW==CUR? net->weights : net->tgt_weights;
  const Activation* const prevStep = agent.Status==1? nullptr : act[0];
  Activation* const currStep = act[1];
  if(agent.Status not_eq 1)
    prevStep->loadMemory(net->mem[agent.ID]);

  const vector<Real> ret = net->predict(inp, prevStep, currStep, W);
  currStep->storeMemory(net->mem[agent.ID]);
  currStep->written = true;
  return ret;
}

vector<Real> Approximator::getOutput(const vector<Real> inp, const int ind,
  Activation*const act, const Uint thrID, const PARAMS USEW) const
{
  //hardcoded to use time series predicted with cur weights for recurrencies:
  const vector<Activation*>& act_cur = series[thrID];
  const Activation*const recur = ind? act_cur[ind-1] : nullptr;
  const Parameters* const W = USEW==CUR? net->weights : net->tgt_weights;
  const vector<Real> ret = net->predict(inp, recur, act, W);
  //if(thrID) cout<<"net fwd with inp:"<<print(inp)<<" out:"<<print(ret)<<endl;
  act->written = true;
  return ret;
}

vector<Real> Approximator::getInput(const Sequence*const traj, const Uint samp, const Uint thrID) const
{
  vector<Real> inp = input->forward(traj, samp, thrID);
  if(relay not_eq nullptr) {
    const vector<Real> addedinp = relay->get(traj, samp, thrID);
    assert(addedinp.size());
    inp.insert(inp.end(), addedinp.begin(), addedinp.end());
  }
  assert(inp.size() == net->getnInputs());
  return inp;
}

void Approximator::backward(vector<Real> error, const Uint samp,
  const Uint thrID, const Uint iSample) const
{
  const Uint netID = thrID + iSample*nThreads;
  gradStats->clip_vector(error);
  gradStats->track_vector(error, thrID);
  const int ind = mapTime2Ind(samp, thrID);
  const vector<Activation*>& act = series[netID];
  assert(act[ind]->written == true && iSample <= extraAlloc);
  //ind+1 because we use c-style for loops in other places: TODO:netID
  error_placements[thrID] = std::max(ind+1, error_placements[thrID]);
  act[ind]->setOutputDelta(error);
}

void Approximator::prepareUpdate()
{
  #pragma omp parallel for //each thread should still handle its own memory
  for(Uint i=0; i<nThreads; i++) if(error_placements[i] > 0) gradient(i);

  if(nAddedGradients == 0) die("Error in prepareUpdate\n");

  opt->prepare_update(nAddedGradients, net->Vgrad);
  nReducedGradients = nAddedGradients;
  nAddedGradients = 0;

  if(mpisize<=1) applyUpdate();
}

void Approximator::applyUpdate()
{
  if(nReducedGradients == 0) return;

  opt->apply_update();
  nReducedGradients = 0;
}

void Approximator::gradient(const Uint thrID) const
{
  if(error_placements[thrID]<=0) return;

  #pragma omp atomic
  nAddedGradients++;

  for(Uint j = 0; j<=extraAlloc; j++)
  {
    const Uint netID = thrID + j*nThreads;
    const vector<Activation*>& act = series[netID];
    const int last_error = error_placements[thrID];

    for (int i=0; i<last_error; i++) assert(act[i]->written == true);

    net->backProp(act, last_error, net->Vgrad[thrID]);

    if(input->net == nullptr) continue;

    for(int i=0; i<last_error; i++) {
      const vector<Real> iG = act[i]->getInputGradient();
      const vector<Real> inputG = vector<Real>(&iG[0], &iG[input->nOutputs()]);
      input->backward(inputG, first_sample[thrID] +i, thrID);
    }
  }
  error_placements[thrID] = -1; //to stop additional backprops
}

void Approximator::getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
{
  gradStats->reduce_stats();
  long double sumW = 0, distTgt = 0;
  net->weights->compute_dist_norm(sumW, distTgt, net->tgt_weights);
  screenOut<<" "<<name<<":[normW:"<<sumW<<" distW:"<<distTgt<<"]";
  fileOut<<" "<<sumW<<" "<<distTgt;
}
