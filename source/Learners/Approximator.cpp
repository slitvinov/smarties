/*
 *  Learner.h
 *  rl
 *
 *  Created by Guido Novati on 15.06.16.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "Approximator.h"

void Aggregator::prepare(const RELAY SET, const Uint thrID) const
{
  usage[thrID] = SET;
}

void Aggregator::prepare_opc(const Sequence*const traj, const Uint samp,
    const Uint thrID) const
{
  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // predict pol/val of all states from samp to T (T-1 if T is term state)
  const Uint nSValues =  traj->tuples.size() - samp - traj->ended;
  const Uint nTotal = nRecurr + nSValues;
  first_sample[thrID] = samp - nRecurr;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nTotal, vector<Real>());
  usage[thrID] = VEC;
}

void Aggregator::prepare_seq(const Sequence*const traj, const Uint thrID) const
{
  const Uint nSValues =  traj->tuples.size() - traj->ended;
  first_sample[thrID] = 0;
  inputs[thrID].clear(); //make sure we only have empty vectors
  inputs[thrID].resize(nSValues, vector<Real>());
  usage[thrID] = VEC;
}

void Aggregator::prepare_one(const Sequence*const traj, const Uint samp,
    const Uint thrID) const
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
  usage[thrID] = VEC;
}

void Aggregator::set(const vector<Real> vec,const Uint samp,const Uint thrID) const
{
  assert(usage[thrID] == VEC);
  const int ind = (int)samp - first_sample[thrID];
  assert(first_sample[thrID] <= (int)samp);
  assert(ind >= 0 && (int) inputs[thrID].size() > ind);
  assert(inputs[thrID][ind].size() == 0);
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
  build.stackSimple( {nInputs}, nouts );
  return build;
}

void Approximator::build_network(Builder& build)
{
  net = build.build();
}

void Approximator::build_finalize(Builder& build)
{
  if(net == nullptr) net = build.build();
  opt = build.finalSimple();

  #pragma omp parallel for
  for (Uint i=0; i<nThreads; i++) // numa aware allocation
   #pragma omp critical
   {
     series[i] = new vector<Activation*>();
     series_tgt[i] = new vector<Activation*>();
     extra_grads[i] = new Grads(net->getnWeights(), net->getnBiases());
   }
  gradStats = new StatsTracker(net->getnOutputs(), name, settings);
}

void Approximator::prepare_opc(const Sequence*const traj, const Uint samp,
    const Uint thrID) const
{
  if(error_placements[thrID] >= 0) gradient(thrID);

  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // predict pol/val of all states from samp to T (T-1 if T is term state)
  const Uint nSValues =  traj->tuples.size() - samp - traj->ended;
  const Uint nTotal = nRecurr + nSValues;
  input->prepare(nTotal, samp - nRecurr, thrID);

  net->prepForBackProp(series[thrID], nTotal);
  net->prepForFwdProp(series_tgt[thrID], nTotal);

  error_placements[thrID] = -1;
  first_sample[thrID] = samp - nRecurr;
}

void Approximator::prepare_seq(const Sequence*const traj, const Uint thrID) const
{
  if(error_placements[thrID] >= 0) gradient(thrID);

  const Uint nSValues =  traj->tuples.size() - traj->ended;
  input->prepare(nSValues, 0, thrID);
  net->prepForBackProp(series[thrID], nSValues);
  net->prepForFwdProp(series_tgt[thrID], nSValues);

  error_placements[thrID] = -1;
  first_sample[thrID] = 0;
}

void Approximator::prepare_one(const Sequence*const traj, const Uint samp,
    const Uint thrID) const
{
  if(error_placements[thrID] >= 0) gradient(thrID);

  // opc requires prediction of some states before samp for recurrencies
  const Uint nRecurr = bRecurrent ? std::min(nMaxBPTT, samp) : 0;
  // might need to predict the value of next state if samp not terminal state
  const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  const Uint nSValues = terminal ? 1 : 2; //probably faster to just assume 2
  const Uint nTotal = nRecurr + nSValues;

  input->prepare(nTotal, samp - nRecurr, thrID);
  net->prepForBackProp(series[thrID], nTotal);
  net->prepForFwdProp(series_tgt[thrID], nTotal);

  error_placements[thrID] = -1;
  first_sample[thrID] = samp - nRecurr;
}

vector<Real> Approximator::forward(const Sequence* const traj, const Uint samp,
  const Uint thrID, const PARAMS USE_WEIGHTS, const PARAMS USE_ACT,
  const int overwrite) const
{
  const auto& act = (USE_ACT==CUR)? *(series[thrID]) : *(series_tgt[thrID]);
  const vector<Activation*>&act_cur = *(series[thrID]);
  const int ind = mapTime2Ind(samp, thrID);

  //if already computed just give answer
  if(act[ind]->written == true && not overwrite)
    return net->getOutputs(act[ind]);

  // write previous outputs if needed (note: will spawn nested function calls)
  if(ind) if(act_cur[ind-1]->written not_eq true)
    this->forward<CUR>(traj, samp-1, thrID);

  const vector<Real> inp = getInput(traj, samp, thrID);
  return getOutput(inp, ind, act[ind], thrID, USE_WEIGHTS);
}

vector<Real> Approximator::relay_backprop(const vector<Real> error,
  const Uint samp, const Uint thrID, const PARAMS USEW) const
{
  const auto& act_tgt = *(series_tgt[thrID]);
  const int ind = mapTime2Ind(samp, thrID), nInp = input->nOutputs();
  assert(act_tgt[ind]->written == true && relay not_eq nullptr);
  net->setOutputDeltas(error, act_tgt[ind]);
  const nnReal*const W=USEW==CUR? net->weights_back : net->tgt_weights_back;
  const nnReal*const B=USEW==CUR? net->biases  :      net->tgt_biases;
  net->backProp(nullptr, act_tgt[ind], nullptr, W, B, extra_grads[thrID]);
  const vector<Real> gradR = net->getInputGradient(act_tgt[ind]);
  return vector<Real>(&gradR[0]+nInp, &gradR[0]+nInp+relay->nOutputs());
}

vector<Real> Approximator::forward_agent(const Sequence* const traj,
  const Agent& agent, const Uint thrID, const PARAMS USEW) const
{
  if(error_placements[thrID] >= 0) gradient(thrID);
  vector<Real> ret(net->getnOutputs());
  const Uint stepid = traj->ndata();
  input->prepare(1, stepid, thrID);
  net->prepForFwdProp(series[thrID], 2);
  const vector<Activation*>& act = *(series[thrID]);
  const vector<Real> inp = getInput(traj, stepid, thrID);
  const nnReal* const W = USEW==CUR ? net->weights : net->tgt_weights;
  const nnReal* const B = USEW==CUR ? net->biases  : net->tgt_biases;

  if(agent.Status==1) net->predict(inp, ret, nullptr, act[1], W, B);
  else { // if i'm using RNN i need to load recur connections (else no effect)
    act[0]->loadMemory(net->mem[agent.ID]); // prevAct[thrID],
    net->predict(inp, ret, act[0], act[1], W, B);
  }
  act[1]->storeMemory(net->mem[agent.ID]);
  return ret;
}

vector<Real> Approximator::getOutput(const vector<Real> inp, const int ind,
  Activation*const act, const Uint thrID, const PARAMS USEW) const
{
  vector<Real> ret(net->getnOutputs());
  //hardcoded to use time series predicted with cur weights for recurrencies:
  const auto& act_cur = *(series[thrID]);
  const nnReal* const W = USEW==CUR ? net->weights : net->tgt_weights;
  const nnReal* const B = USEW==CUR ? net->biases  : net->tgt_biases;
  net->predict(inp, ret, (ind ? act_cur[ind-1] : nullptr), act, W, B);
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
  const Uint thrID)const
{
  gradStats->track_vector(error, thrID);
  gradStats->clip_vector(error);
  const int ind = mapTime2Ind(samp, thrID);
  const vector<Activation*>& act = *(series[thrID]);
  assert(act[ind]->written == true);
  //ind+1 because we use c-style for loops in other places:
  error_placements[thrID] = std::max(ind+1, error_placements[thrID]);

  net->setOutputDeltas(error, act[ind]);
}

void Approximator::update()
{
  if(!nAddedGradients) die("Error in stackAndUpdateNNWeights\n");

  #pragma omp parallel for //each thread should still handle its own memory
  for(Uint i=0; i<nThreads; i++) if(error_placements[i] >= 0) gradient(i);

  opt->nepoch++;
  Uint nTotGrads = nAddedGradients;
  opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads
  if (learn_size > 1)
  { //add up gradients across masters
    MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
        MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
        MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
    MPI_Allreduce(MPI_IN_PLACE,&nTotGrads,1,MPI_UNSIGNED,MPI_SUM,mastersComm);
  }
  //update is deterministic: can be handled independently by each node
  //communication overhead is probably greater than a parallelised sum
  opt->update(net->grad, nTotGrads);

  if(tgtUpdateAlpha > 0) {
    if (cntUpdateDelay == 0) { //DQN-style frozen weight
      cntUpdateDelay = tgtUpdateAlpha;
      opt->moveFrozenWeights(tgtUpdateAlpha);
    }
    if(cntUpdateDelay>0) cntUpdateDelay--;
  }
}

void Approximator::gradient(const Uint thrID) const
{
  if(error_placements[thrID]<0) return;

  #pragma omp atomic
  nAddedGradients++;

  vector<Activation*>& act = *(series[thrID]);
  if (thrID==0) net->backProp(act, error_placements[thrID], net->grad);
  else net->backProp(act, error_placements[thrID], net->Vgrad[thrID]);
  error_placements[thrID] = -1; //to stop additional backprops

  if(input->net == nullptr) return;

  for(int i=0; i<error_placements[thrID]; i++) {
    const vector<Real> grad0 = net->getInputGradient(act[i]);
    const Uint inpFeat = input->nOutputs();
    const vector<Real> inpgrad = vector<Real>(&grad0[0], &grad0[0] + inpFeat);
    input->backward(inpgrad, first_sample[thrID] + i, thrID);
    net->addOutputDeltas(inpgrad, act[i]);
  }
}

void Approximator::getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
{
  gradStats->reduce_stats();
  long double sumW = 0, distTgt = 0;
  #pragma omp parallel for reduction(+:sumW,distTgt)
  for (Uint w=0; w<net->getnWeights(); w++) {
    sumW += std::fabs(net->weights[w]);
    //sumWSq += net->weights[w]*net->weights[w];
    distTgt += std::fabs(net->weights[w]-net->tgt_weights[w]);
  }
  screenOut<<" "<<name<<":[normW:"<<sumW<<" distW:"<<distTgt<<"]";
  fileOut<<" "<<sumW<<" "<<distTgt;
}
