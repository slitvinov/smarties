/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "NAF.h"

NAF::NAF(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm,_env,settings,settings.nnOutputs),
nA(_env->aI.dim), nL(compute_nL(_env->aI.dim))
{
  #ifdef NDEBUG
  //if(bRecurrent) die("NAF recurrent not tested!\n");
  #endif

  //#ifdef FEAT_CONTROL
  //const Uint task_out0 = ContinuousSignControl::addRequestedLayers(nA,
  //   env->sI.dimUsed, net_indices, net_outputs, out_weight_inits);
  //#endif

  buildNetwork(net_outputs, settings);

  printf("NAF: Built network with outputs: %s %s\n",
    print(net_indices).c_str(), print(net_outputs).c_str());
  assert(nOutputs == net->getnOutputs());
  assert(nInputs == net->getnInputs());
  policyVecDim = 2*nA;
  //#ifdef FEAT_CONTROL
  //task = new ContinuousSignControl(task_out0, nA, env->sI.dimUsed, net, data);
  //#endif
  test();
}

void NAF::select(const int agentId, const Agent& agent)
{
  vector<Real> beta(policyVecDim,0);
  if(agent.Status==2) { data->passData(agentId, agent, beta); return; }

  vector<Real> output = output_value_iteration(agentId, agent);
  const Quadratic_advantage advantage = prepare_advantage(output);
  //load computed policy into a
  vector<Real> policy = advantage.getMean();
  const Real annealedVar = bTrain ? .2*annealingFactor()+greedyEps : greedyEps;

  if(positive(annealedVar)) {
    std::normal_distribution<Real> dist(0, annealedVar);
    for(Uint i=0; i<nA; i++) {
      beta[i] = policy[i];
      beta[i+nA] = 1/annealedVar/annealedVar;
      policy[i] += dist(*gen);
    }
  }

  //scale back to action space size:
  agent.a->set(aInfo.getScaled(policy));
  data->passData(agentId, agent, beta);
  dumpNetworkInfo(agentId);
}

void NAF::Train_BPTT(const Uint seq, const Uint thrID) const
{
  const Real rGamma = annealedGamma();
  const Uint ndata = data->Set[seq]->tuples.size();
  const Uint nValues = data->Set[seq]->ended ? ndata-1 :ndata;
  vector<Activation*> actcur = net->allocateUnrolledActivations(ndata-1);
  vector<Activation*> acthat = net->allocateUnrolledActivations(nValues);

  for (Uint k=0; k<nValues; k++) {
    const vector<Real> inp = data->standardize(data->Set[seq]->tuples[k]->s);
    if(k<ndata-1) net->seqPredict_inputs(inp, actcur[k]);
    net->seqPredict_inputs(inp, acthat[k]);
  }
  net->seqPredict_execute(actcur,actcur);
  net->seqPredict_execute(actcur,acthat,net->tgt_weights,net->tgt_biases);

  for (Uint k=0; k<ndata-1; k++) { //state in k=[0:N-2]
    const bool term = k+2==ndata && data->Set[seq]->ended;
    const Tuple * const _t = data->Set[seq]->tuples[k+1]; //contains sNew, rew
    const Tuple * const t_ = data->Set[seq]->tuples[k]; //contains sOld, act
    const vector<Real>output = net->getOutputs(actcur[k]);
    const vector<Real>target =term?vector<Real>():net->getOutputs(acthat[k+1]);

    const Real Vsold = output[net_indices[0]];
    const vector<Real> act = aInfo.getInvScaled(t_->a); //unbounded action space
    const Quadratic_advantage adv_sold = prepare_advantage(output);
    const Real Qsold = Vsold + adv_sold.computeAdvantage(act);
    const Real value = (term) ? _t->r : _t->r + rGamma*target[net_indices[0]];
    const Real error = value - Qsold;
    vector<Real> gradient(nOutputs);
    gradient[net_indices[0]] = error;
    adv_sold.grad(act, error, gradient);

    //#ifdef FEAT_CONTROL
    //  const Activation* const recur = term ? nullptr : acthat[k+1];
    //  task->Train(actcur[k], recur, act, seq, k, rGamma, gradient);
    //#endif

    statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], gradient);
    clip_gradient(gradient, stdGrad[0], seq, k);
    dumpStats(Vstats[thrID], Qsold, error);
    data->Set[seq]->tuples[k]->SquaredError = error*error;
    net->setOutputDeltas(gradient, actcur[k]);
  }

  if (thrID==0) net->backProp(actcur, net->grad);
  else net->backProp(actcur, net->Vgrad[thrID]);
  net->deallocateUnrolledActivations(&actcur);
  net->deallocateUnrolledActivations(&acthat);
}

void NAF::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  const Real rGamma = annealedGamma();
  vector<Real> target(nOutputs), gradient(nOutputs, 0);
  const Uint ndata = data->Set[seq]->tuples.size(), nMaxBPTT = MAX_UNROLL_BFORE;
  const Uint iRecurr = bRecurrent ? max(nMaxBPTT,samp)-nMaxBPTT : samp;
  const Uint nRecurr = bRecurrent ? min(nMaxBPTT,samp)+1         : 1;
  const bool terminal = samp+2==ndata && data->Set[seq]->ended;
  vector<Activation*> series_cur = net->allocateUnrolledActivations(nRecurr);
  Activation* tgtAct = terminal ? nullptr : net->allocateActivation();
  //if(thrID==1) { printf("%d %u %u %u %u %u\n",terminal,seq,samp,ndata,iRecurr,nRecurr); fflush(0); }

  if(thrID==1) profiler->stop_start("FWD");

  for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
    assert((k==samp)==(j==nRecurr-1));
    const Tuple * const _t = data->Set[seq]->tuples[k];
    net->seqPredict_inputs(data->standardize(_t->s), series_cur[j]);
  }
  //all are loaded: execute the whole loop:
  net->seqPredict_execute(series_cur, series_cur);

  if(thrID==1)  profiler->stop_start("CMP");

  //extract the only output we actually correct:
  const vector<Real> output = net->getOutputs(series_cur.back());
  const Tuple* const t_ = data->Set[seq]->tuples[samp];
  const Tuple* const _t = data->Set[seq]->tuples[samp+1];
  const vector<Real> act = aInfo.getInvScaled(t_->a); //unbounded action
  const Quadratic_advantage adv_sold = prepare_advantage(output);

  if (not terminal)
    net->predict(data->standardize(_t->s), target, series_cur.back(), tgtAct, net->tgt_weights, net->tgt_biases);

  const Real Vsold = output[net_indices[0]], Vsnew = target[net_indices[0]];
  const Real Qsold = Vsold + adv_sold.computeAdvantage(act);
  const Real value = (terminal) ? _t->r : _t->r + rGamma*Vsnew;
  const Real error = value - Qsold;
  gradient[net_indices[0]] = error;
  adv_sold.grad(act, error, gradient);

  //#ifdef FEAT_CONTROL
  //  task->Train(series_cur.back(),tgtAct,act,seq,samp,rGamma,gradient);
  //#endif

  statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], gradient);
  data->Set[seq]->tuples[samp]->SquaredError = error*error;
  clip_gradient(gradient, stdGrad[0], seq, samp);
  dumpStats(Vstats[thrID], Qsold, error);
  net->setOutputDeltas(gradient, series_cur.back());

  if(thrID==1)  profiler->stop_start("BCK");

  if (thrID==0) net->backProp(series_cur, net->grad);
  else net->backProp(series_cur, net->Vgrad[thrID]);
  net->deallocateUnrolledActivations(&series_cur);
  _dispose_object(tgtAct);

  if(thrID==1)  profiler->stop_start("TSK");
}
