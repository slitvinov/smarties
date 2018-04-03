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

void Aggregator::prepare(const Uint N, const Sequence*const traj,
  const Uint samp, const Uint thrID, const RELAY SET) const
{
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  const Uint nTotal = nRecurr + N;
  first_sample[thrID] = samp - nRecurr;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, Rvec());
  usage[thrID] = SET;
}

void Aggregator::prepare_seq(const Sequence*const traj, const Uint thrID, const RELAY SET) const
{
  const Uint nSValues =  traj->tuples.size() - traj->ended;
  first_sample[thrID] = 0;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nSValues, Rvec());
  usage[thrID] = SET;
}

void Aggregator::prepare_one(const Sequence*const traj, const Uint samp,
    const Uint thrID, const RELAY SET) const
{
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const Uint nSValues = traj->isTerminal(samp+1) ? 1 : 2;
  const Uint nTotal = nRecurr + nSValues;
  first_sample[thrID] = samp - nRecurr;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, Rvec());
  usage[thrID] = SET;
}

void Aggregator::set(const Rvec vec,const Uint samp,const Uint thrID) const
{
  usage[thrID] = VEC;
  const int ind = (int)samp - first_sample[thrID];
  assert(first_sample[thrID] <= (int)samp);
  assert(ind >= 0 && (int) inputs[thrID].size() > ind);
  inputs[thrID][ind] = vec;
}

Rvec Aggregator::get(const Sequence*const traj, const Uint samp,
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

void Approximator::initializeNetwork(Builder& build, Real cutGradFactor)
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
      relayG[i] = net->allocateParameters();
   }

  if(relay not_eq nullptr) {
    vector<int> relayInputID;
    for(Uint i=1; i<net->layers.size(); i++) //assume layer 0 is passed to input
      if(net->layers[i]->bInput) relayInputID.push_back(i);

    if(relayInputID.size() > 1) { die("should not be possible");
    } else if (relayInputID.size() == 1) {
      relayInp = relayInputID[0];
      if(net->layers[relayInp]->nOutputs() != relay->nOutputs()) die("crap");
    } else relayInp = 0;
  }
  #ifdef __CHECK_DIFF //check gradients with finite differences
    net->checkGrads();
  #endif
  gradStats=new StatsTracker(net->getnOutputs(),name+"_grads",settings,cutGradFactor);
}

void Approximator::prepare(const Uint N, const Sequence*const traj,
  const Uint samp, const Uint thrID, const Uint nSamples) const
{
  if(error_placements[thrID] > 0) gradient(thrID);
  assert(nSamples<=1+extraAlloc && nSamples>0);
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  const Uint nTotal = nRecurr + N;
  input->prepare(nTotal, samp - nRecurr, thrID);

  for(Uint k=0; k<nSamples; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  if(series_tgt.size()>thrID)
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
  const Uint nSValues = traj->isTerminal(samp+1) ? 1 : 2;
  const Uint nTotal = nRecurr + nSValues;

  input->prepare(nTotal, samp - nRecurr, thrID);
  for(Uint k=0; k<nSamples; k++)
    net->prepForBackProp(series[thrID + k*nThreads], nTotal);

  net->prepForFwdProp(series_tgt[thrID], nTotal);

  error_placements[thrID] = -1;
  first_sample[thrID] = samp - nRecurr;
}

Rvec Approximator::forward(const Sequence* const traj, const Uint samp,
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

  const Rvec inp = getInput(traj, samp, thrID);
  //cout <<"Input : "<< print(inp) << endl; fflush(0);
  return getOutput(inp, ind, act[ind], thrID, USE_WEIGHTS);
}

Rvec Approximator::relay_backprop(const Rvec err,
  const Uint samp, const Uint thrID, const PARAMS USEW) const
{
  if(relay == nullptr || relayInp < 0) die("improperly set up the relay");
  const vector<Activation*>& act = series_tgt[thrID];
  const int ind = mapTime2Ind(samp, thrID), nInp = input->nOutputs();
  assert(act[ind]->written == true && relay not_eq nullptr);
  const Parameters*const W = USEW==CUR? net->weights : net->tgt_weights;
  const Rvec ret = net->inpBackProp(err, act[ind], relayG[thrID], W, relayInp);
  if(relayInp>0) return ret;
  else return Rvec(&ret[nInp], &ret[nInp+relay->nOutputs()]);
}

Rvec Approximator::forward_agent(const Sequence* const traj,
  const Agent& agent, const Uint thrID, const PARAMS USEW) const
{
  if(error_placements[thrID] > 0) gradient(thrID);

  const Uint stepid = traj->ndata();
  input->prepare(1, stepid, thrID);
  net->prepForFwdProp(series[thrID], 2);

  const vector<Activation*>& act = series[thrID];
  const Rvec inp = getInput(traj, stepid, thrID);
  const Parameters* const W = USEW==CUR? net->weights : net->tgt_weights;
  const Activation* const prevStep = agent.Status==INIT_COMM? nullptr : act[0];
  act[0]->written = true; act[1]->written = true;
  Activation* const currStep = act[1];
  if(agent.Status not_eq INIT_COMM) prevStep->loadMemory(net->mem[agent.ID]);
  const Rvec ret = net->predict(inp, prevStep, currStep, W);
  currStep->storeMemory(net->mem[agent.ID]);
  return ret;
}

Rvec Approximator::getOutput(const Rvec inp, const int ind,
  Activation*const act, const Uint thrID, const PARAMS USEW) const
{
  //hardcoded to use time series predicted with cur weights for recurrencies:
  const vector<Activation*>& act_cur = series[thrID];
  const Activation*const recur = ind? act_cur[ind-1] : nullptr;
  const Parameters* const W = USEW==CUR? net->weights : net->tgt_weights;
  const Rvec ret = net->predict(inp, recur, act, W);
  //if(thrID) cout<<"net fwd with inp:"<<print(inp)<<" out:"<<print(ret)<<endl;
  act->written = true;
  return ret;
}

Rvec Approximator::getInput(const Sequence*const traj, const Uint samp, const Uint thrID) const
{
  Rvec inp = input->forward(traj, samp, thrID);
  if(relay not_eq nullptr) {
    const Rvec addedinp = relay->get(traj, samp, thrID);
    assert(addedinp.size());
    inp.insert(inp.end(), addedinp.begin(), addedinp.end());
  }
  assert(inp.size() == net->getnInputs());
  return inp;
}

void Approximator::backward(Rvec error, const Uint samp,
  const Uint thrID, const Uint iSample) const
{
  const Uint netID = thrID + iSample*nThreads;
  gradStats->track_vector(error, thrID);
  gradStats->clip_vector(error);
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

  if(nAddedGradients == 0) warn("Zero-gradient update. Revise hyperparameters.\n");

  opt->prepare_update(nAddedGradients, net->Vgrad);
  reducedGradients = 1;
  nAddedGradients = 0;

  if(mpisize<=1) applyUpdate();
}

void Approximator::applyUpdate()
{
  if(reducedGradients == 0) return;

  opt->apply_update();
  reducedGradients = 0;
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

    if(input->net == nullptr || blockInpGrad) continue;

    for(int i=0; i<last_error; i++) {
      Rvec inputG = act[i]->getInputGradient(0);
      inputG.resize(input->nOutputs());
      input->backward(inputG, first_sample[thrID] +i, thrID);
    }
  }
  error_placements[thrID] = -1; //to stop additional backprops
}

void Approximator::getHeaders(ostringstream& buff) const
{
  buff << std::left << std::setfill(' ') ;
  buff <<"| " << std::setw(6) << name << ":|W| DW cut%";
}

void Approximator::getMetrics(ostringstream& buff) const
{
  long double sumW = 0, distTgt = 0;
  net->weights->compute_dist_norm(sumW, distTgt, net->tgt_weights);
  buff<<" "<<std::setw(6)<<std::setprecision(0)<<sumW;
  buff<<" "<<std::setw(6)<<std::setprecision(0)<<distTgt;
  buff<<" "<<std::setw(5)<<std::setprecision(3)<<gradStats->clip_ratio()*100;
}
