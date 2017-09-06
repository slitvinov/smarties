/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#include "../StateAction.h"
#include "GAE.h"
#define PPO_PENALKL
#define PPO_CLIPPED

GAE::GAE(MPI_Comm comm, Environment*const _env, Settings & settings) :
Learner_onPolicy(comm, _env, settings, settings.nnOutputs)
{
  settings.splitLayers = 9; // all!
  Builder build(settings);
  vector<Uint> inpv {nInputs}, outs {nA, 1};
  pair<Uint,Real> stdv = make_pair(nA, -2*std::log(greedyEps));
  vector<Real> onev {1/settings.klDivConstraint};
  build.buildSimple(net, opt, inpv, outs, stdv, onev);

  for (Uint i = 0; i < nThreads; i++) {
    series_1.push_back(new vector<Activation*>());
    series_2.push_back(new vector<Activation*>());
  }
  if(currAct.size() < nThreads ) net->prepForFwdProp(&currAct, nThreads);
  if(prevAct.size() < nThreads ) net->prepForFwdProp(&prevAct, nThreads);

  printf("GAE: Built network with outputs: %s %s\n",
    print(net_indices).c_str(),print(net_outputs).c_str());
}

void GAE::select(const int agentId, const Agent& agent)
{
  const int thrID = omp_get_thread_num();
  const int workid = retrieveAssignment(agentId);
  Real fac_lambda = lambda*gamma, fac_gamma = gamma;
  //printf("Thread %d working with agent %d on task %d with status %d\n", thrid, agentId, workid, agent.Status);
  //fflush(0);
  if(workid<0) die("workspace not allocated.");
  //printf("(%lu %lu %lu)\n", work[workid]->series.size(), work[workid]->actions.size(), work[workid]->rewards.size());
  //fflush(0);

  if(agent.Status==2) //terminal state
  {
    data->writeData(agentId, agent, vector<Real>(2*nA,0));
    if(work[workid]->Vst.size() == 0) //empty trajectory
    {
      work[workid]->clear();
      return;
    }
    // V_s_term := 0, therefore delta_term = r_t+1 - V(s_t)
    const Real delta = agent.r - work[workid]->Vst.back();
    work[workid]->rewards.push_back(agent.r);
    work[workid]->GAE.push_back(delta);

    assert(work[workid]->GAE.size() == work[workid]->rewards.size());
    assert(work[workid]->GAE.size() == work[workid]->Vst.size());
    for (Uint i=2; i<=work[workid]->GAE.size(); i++) {
      const Uint ind = work[workid]->GAE.size() - i;
      work[workid]->rewards[ind] += fac_gamma*agent.r; //sum of disc. rews
      work[workid]->GAE[ind] += fac_lambda*delta; // sum of lambda-corrections
      fac_lambda *= lambda*gamma; fac_gamma *= gamma; //prep next
    }
    work[workid]->done = 1;
    addTasks(work[workid]);
    return;
  }

  if(thrID==1) profiler->stop_start("FWD");
  if(thrID==0) profiler_ext->stop_start("WORK");

  vector<Real> output(nOutputs);
  const vector<Real> input = data->standardize(agent.s->copy_observed());
  //if required, chain together nAppended obs to compose state
  assert(!nAppended); //not supported

  net->predict(input, output, currAct[thrID]);

  const Real val = output[ValID];
  const auto pol = prepare_policy(output);
  vector<Real> beta_mean=pol.getMean(), beta_std=pol.getStdev(), beta(2*nA,0);
  vector<Real> act(nA,0);
  for(Uint i=0; i<nA; i++) {
    beta[i] = beta_mean[i];
    beta[nA+i] = beta_std[i];
    //#ifdef INTEGRATEANDFIREMODEL
    //std::lognormal_distribution<Real> dist_cur(beta_mean[i], beta_std[i]);
    //#else
    std::normal_distribution<Real> dist_cur(beta_mean[i], beta_std[i]);
    //#endif
    act[i] = bTrain ? dist_cur(*gen) : beta_mean[i];
  }

  //treated as first in two circumstances:
  // - if actually initial state
  // - or if policy was updated after prev action
  // (in PPO T_horizon is not linked to Term. states)
  const bool first = agent.Status==1 || work[workid]->Vst.size() == 0;
  if(not first)
  {
    // delta_t = r_t+1 + gamma V(s_t+1) - V(s_t)  (pedix on r means r_t+1
    // received with transition to s_t+1, sometimes referred to as r_t)
    const Real delta = agent.r +gamma*val -work[workid]->Vst.back();
    work[workid]->rewards.push_back(agent.r);
    work[workid]->GAE.push_back(delta);
    for (Uint i=2; i<=work[workid]->GAE.size(); i++) {
      const Uint ind = work[workid]->GAE.size() - i;
      work[workid]->rewards[ind] += fac_gamma*agent.r;
      work[workid]->GAE[ind] += fac_lambda*delta;
      fac_lambda *= lambda*gamma; fac_gamma *= gamma;
    }
  }

  work[workid]->Vst.push_back(val);
  assert(work[workid]->GAE.size()+1 == work[workid]->Vst.size());

  work[workid]->actions.push_back(act);
  work[workid]->policy.push_back(beta);
  work[workid]->observations.push_back(input);

  //#ifndef INTEGRATEANDFIREMODEL
  agent.a->set(aInfo.getScaled(act));
  //#else
  //agent.a->set(act);
  //#endif

  data->writeData(agentId, agent, beta);
  if(thrID==0) profiler_ext->stop_start("COMM");
  if(thrID==1) profiler->pop_stop();
  //data->passData(agentId, agent, beta);
  //dumpNetworkInfo(agentId);
}

void GAE::Train(const Uint workid, const Uint samp, const Uint thrID) const
{
  if(thrID==1)  profiler->stop_start("TRAIN");

  vector<Real> output(nOutputs), grad(nOutputs,0);
  const vector<Real> act = completed[workid]->actions[samp];
  const Real val_tgt = completed[workid]->rewards[samp];
  const Real adv_est = completed[workid]->GAE[samp];
    //printf("%u %u %u %f %f \n", workid, samp, thrID, val_tgt, adv_est);
  net->prepForBackProp(series_1[thrID], 1);
  vector<Activation*>& series = *(series_1[thrID]);
  const vector<Real>& input = completed[workid]->observations[samp];
  net->predict(input, output, series, 0);

  const Real Vst = output[ValID];
  const auto pol = prepare_policy(output);
  const auto pol_hat = prepare_behavior(completed[workid]->policy[samp]);
  const Real actProbOnPolicy = pol.evalLogProbability(act);
  const Real actProbBehavior = pol_hat.evalLogProbability(act);
  const Real rho_cur = min(MAX_IMPW,safeExp(actProbOnPolicy-actProbBehavior));
  const Real DivKL=pol.kl_divergence_opp(&pol_hat), penalDKL=output[PenalID];

  Real gain = rho_cur*adv_est;
  #ifdef PPO_CLIPPED
    if (adv_est > 0 && rho_cur > 1+clip_fac) gain = 0;
    if (adv_est < 0 && rho_cur < 1-clip_fac) gain = 0;
  #endif

  #ifdef PPO_PENALKL
    const vector<Real> policy_grad = pol.policy_grad(act, gain);
    const vector<Real> penal_grad = pol.div_kl_opp_grad(&pol_hat, -penalDKL);
    vector<Real> totalPolGrad = sum2Grads(penal_grad, policy_grad);
  #else //we still learn the penal coef, for simplicity, but no effect
    if(gain==0) {
      if(thrID==1)  profiler->stop_start("SLP");
      return; //if 0 pol grad dont backprop
    }
    vector<Real> totalPolGrad = pol.policy_grad(act, gain);
  #endif

  grad[ValID] = val_tgt - Vst;
  pol.finalize_grad(totalPolGrad, grad, aInfo.bounded);
  grad[PenalID] = 4*std::pow(DivKL - DKL_target,3)*penalDKL;

  //bookkeeping:
  dumpStats(Vstats[thrID], Vst, val_tgt - Vst);
  statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
  clip_grad(grad, stdGrad[0]);
  net->setOutputDeltas(grad, series.back());

  if(thrID==1)  profiler->stop_start("BCK");
  if (thrID==0) net->backProp(series, net->grad);
  else net->backProp(series, net->Vgrad[thrID]);
  if(thrID==1)  profiler->stop_start("SLP");
}

void GAE::Train_BPTT(const Uint workid, const Uint thrID) const
{
  die("not allowed");
}
