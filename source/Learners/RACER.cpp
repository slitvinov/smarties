/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "RACER.h"
//#include "RACER_TrainBPTT.cpp"
//#include "RACER_Train.cpp"
//#define DUMP_EXTRA

RACER::RACER(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_utils(comm,_env,settings,settings.nnOutputs),
truncation(5), delta(1), nA(_env->aI.dim), nL(compute_nL(_env->aI.dim)),
generators(settings.generators)
{
  vector<Real> out_weight_inits = {-1, -1, settings.outWeightsPrefac};
  #ifndef ACER_SAFE
  out_weight_inits.push_back(-1);
  #endif
  #ifndef ACER_RELAX
  out_weight_inits.push_back(settings.outWeightsPrefac);
  #endif
  #ifdef FEAT_CONTROL
    const Uint task_out0 = ContinuousSignControl::addRequestedLayers(nA,
       env->sI.dimUsed, net_indices, net_outputs, out_weight_inits);
  #endif

  buildNetwork(net, opt, net_outputs, settings, out_weight_inits);
  printf("RACER: Built network with outputs: %s %s\n",
    print(net_indices).c_str(),print(net_outputs).c_str());
  assert(nOutputs == net->getnOutputs());
  assert(nInputs == net->getnInputs());
  #ifdef FEAT_CONTROL
  task = new ContinuousSignControl(task_out0, nA, env->sI.dimUsed, net, data);
  #endif
  #ifdef DUMP_EXTRA
    policyVecDim = 3*nA + nL;
  #else
    policyVecDim = 2*nA;
  #endif
  //data->bRecurrent =bRecurrent =true;
  //data->bRecurrent =bRecurrent =false;
  test();
}

void RACER::select(const int agentId, const Agent& agent)
{
  if(agent.Status==2) { //no need for action, just pass terminal s & r
    data->passData(agentId,agent,vector<Real>(policyVecDim,0));
    return;
  }
  vector<Real> output = output_stochastic_policy(agentId, agent);
  assert(output.size() == nOutputs);
  //variance is pos def: transform linear output layer with softplus

  const Gaussian_policy pol = prepare_policy(output);
  const Quadratic_advantage adv = prepare_advantage(output, &pol);
  const Real anneal = annealingFactor();
  const Real safety_std = std::sqrt(1/ACER_MAX_PREC);
  vector<Real> beta_mean=pol.getMean(), beta_std=pol.getStdev(), beta(2*nA,0);

  if(bTrain)
  for(Uint i=0; i<nA; i++) {
    beta_std[i] = std::max(safety_std +  anneal*greedyEps, beta_std[i]);
    //beta_mean[i] = (1-anneal*anneal)*beta_mean[i];
  }

  vector<Real> act = positive(greedyEps+anneal) ? Gaussian_policy::sample(gen, beta_mean, beta_std) : beta_mean;

  //scale back to action space size:
  agent.a->set(aInfo.getScaled(act));

  #ifdef DUMP_EXTRA
  beta.insert(beta.end(), adv.matrix.begin(), adv.matrix.end());
  beta.insert(beta.end(), adv.mean.begin(),   adv.mean.end());
  #endif
  data->passData(agentId, agent, beta);
  dumpNetworkInfo(agentId);
}

void RACER::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  //this should go to gamma rather quick:
  const Real rGamma = annealedGamma();
  const Uint ndata = data->Set[seq]->tuples.size();
  assert(samp<ndata-1);
  const bool bEnd = data->Set[seq]->ended; //whether sequence has terminal rew
  const Uint nMaxTargets = MAX_UNROLL_AFTER+1, nMaxBPTT = MAX_UNROLL_BFORE;
  //for off policy correction we need reward, therefore not last one:
  Uint nSUnroll = min(                     ndata-samp-1, nMaxTargets-1);
  //if we do not have a terminal reward, then we compute value of last state:
  Uint nSValues = min(bEnd? ndata-1-samp : ndata-samp  , nMaxTargets  );
  //if truncated seq, we cannot compute the OFFPOL correction for the last one
  const Uint nRecurr = bRecurrent ? min(nMaxBPTT,samp)+1        : 1;
  const Uint iRecurr = bRecurrent ? max(nMaxBPTT,samp)-nMaxBPTT : samp;
  //if(thrID==1) { printf("%d %u %u %u %u %u %u\n", bEnd, samp, ndata, nSUnroll, nSValues, nRecurr, iRecurr); fflush(0); }
  if(thrID==1) profiler->stop_start("FWD");

  vector<vector<Real>> out_cur(1, vector<Real>(nOutputs,0));
  vector<vector<Real>> out_hat(nSValues, vector<Real>(nOutputs,0));
  vector<Activation*> series_cur = net->allocateUnrolledActivations(nRecurr);
  vector<Activation*> series_hat = net->allocateUnrolledActivations(nSValues);

  for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
    const vector<Real> inp = data->standardize(data->Set[seq]->tuples[k]->s);
    net->seqPredict_inputs(inp, series_cur[j]);
    if(k==samp) { //all are loaded: execute the whole loop:
      assert(j==nRecurr-1);
      net->seqPredict_execute(series_cur, series_cur);
      //extract the only output we actually correct:
      net->seqPredict_output(out_cur[0], series_cur[j]); //the humanity!
      //predict samp with target weight using curr recurrent inputs as estimate:
      const Activation*const recur = j ? series_cur[j-1] : nullptr;
      net->predict(inp, out_hat[0], recur, series_hat[0], net->tgt_weights, net->tgt_biases);
    }
  }

  Real importanceW = 1, C = ACER_LAMBDA*rGamma;
  for(Uint k=1; k<nSValues; k++)
  {
    net->predict(data->standardized(seq, k+samp), out_hat[k], series_hat, k
    #ifndef ACER_AGGRESSIVE
      , net->tgt_weights, net->tgt_biases
    #endif
    );

    #ifndef NO_CUT_TRACES
    if (k == nSValues-1) break;
    const Tuple* const _t = data->Set[seq]->tuples[k+samp];
    //else check if the importance weight is too small to continue:
    const Gaussian_policy pol_hat = prepare_policy(out_hat[k]);
    const vector<Real> act = aInfo.getInvScaled(_t->a);//unbounded action space
    const Real probTrgt = pol_hat.evalLogProbability(act);
    const Real probBeta = Gaussian_policy::evalBehavior(act, _t->mu);
    importanceW *= C * std::min(truncation, safeExp(probTrgt-probBeta));
    if (importanceW < std::numeric_limits<Real>::epsilon()) {
      //printf("Cut trace afert %u out of %u samples!\n",k,nSValues);
      nSUnroll = k; //for this last state we do not compute offpol correction
      nSValues = k+1; //we initialize value of Q_RET to V(state)
      break;
    }
    #endif
  }

  if(thrID==1)  profiler->stop_start("ADV");

  Real Q_RET = 0, Q_OPC = 0;
  if(nSValues != nSUnroll) //partial sequence: compute value of term state
    Q_RET=Q_OPC= out_hat[nSValues-1][net_indices[0]]; //V(s_T) with tgt weights

  for (int k=static_cast<int>(nSUnroll)-1; k>0; k--) //propagate Q to k=0
    offPolCorrUpdate(seq, k+samp, Q_RET, Q_OPC, out_hat[k], rGamma);

  if(thrID==1)  profiler->stop_start("CMP");

  vector<Real> grad = compute<0>(seq, samp, Q_RET, Q_OPC, out_cur[0], out_hat[0], rGamma, thrID);

  #ifdef FEAT_CONTROL
    const vector<Real> act=aInfo.getInvScaled(data->Set[seq]->tuples[samp]->a);
    const Activation*const recur = nSValues>1 ? series_hat[1] : nullptr;
    task->Train(series_cur.back(), recur, act, seq, samp, rGamma, grad);
  #endif

  //write gradient onto output layer:
  statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
  clip_gradient(grad, stdGrad[0], seq, samp);
  net->setOutputDeltas(grad, series_cur.back());

  if(thrID==1)  profiler->stop_start("BCK");

  if (thrID==0) net->backProp(series_cur, net->grad);
  else net->backProp(series_cur, net->Vgrad[thrID]);
  net->deallocateUnrolledActivations(&series_cur);
  net->deallocateUnrolledActivations(&series_hat);

  if(thrID==1)  profiler->pop_stop();
}

void RACER::Train_BPTT(const Uint seq, const Uint thrID) const
{
  //this should go to gamma rather quick:
  const Real rGamma = annealedGamma();
  const Uint ndata = data->Set[seq]->tuples.size();
  vector<Activation*> series_cur = net->allocateUnrolledActivations(ndata-1);
  vector<Activation*> series_hat = net->allocateUnrolledActivations(ndata-1);

  if(thrID==1) profiler->stop_start("FWD");

  for (Uint k=0; k<ndata-1; k++) {
    const Tuple * const _t = data->Set[seq]->tuples[k]; // s, a, mu
    const vector<Real> scaledSold = data->standardize(_t->s);
    //const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
    net->seqPredict_inputs(scaledSold, series_cur[k]);
    net->seqPredict_inputs(scaledSold, series_hat[k]);
  }
  net->seqPredict_execute(series_cur,series_cur);
  net->seqPredict_execute(series_cur,series_hat,net->tgt_weights,net->tgt_biases);

  if(thrID==1)  profiler->stop_start("CMP");

  Real Q_RET = 0, Q_OPC = 0;
  //if partial sequence then compute value of last state (!= R_end)
  if(not data->Set[seq]->ended) {
    series_hat.push_back(net->allocateActivation());
    const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
    vector<Real> out_T(nOutputs, 0), S_T = data->standardize(_t->s);//last state
    net->predict(S_T, out_T, series_hat, ndata-1
    #ifndef ACER_AGGRESSIVE
      , net->tgt_weights, net->tgt_biases
    #endif
    );
    Q_OPC = Q_RET = out_T[net_indices[0]]; //V(s_T) computed with tgt weights
  }

  for (int k=static_cast<int>(ndata)-2; k>=0; k--)
  {
    vector<Real> out_cur = net->getOutputs(series_cur[k]);
    vector<Real> out_hat = net->getOutputs(series_hat[k]);
    vector<Real> grad = compute<1>(seq, k, Q_RET, Q_OPC, out_cur, out_hat, rGamma, thrID);
    #ifdef FEAT_CONTROL
    const vector<Real> act=aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
    task->Train(series_cur[k], series_hat[k+1], act, seq, k, rGamma, grad);
    #endif

    //write gradient onto output layer:
    statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
    clip_gradient(grad, stdGrad[0], seq, k);
    net->setOutputDeltas(grad, series_cur[k]);
  }

  if(thrID==1)  profiler->stop_start("BCK");

  if (thrID==0) net->backProp(series_cur, net->grad);
  else net->backProp(series_cur, net->Vgrad[thrID]);
  net->deallocateUnrolledActivations(&series_cur);
  net->deallocateUnrolledActivations(&series_hat);

  if(thrID==1)  profiler->pop_stop();
}
