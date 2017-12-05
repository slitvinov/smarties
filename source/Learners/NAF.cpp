/*
 *  NAF.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#include "../StateAction.h"
#include "../Network/Builder.h"
#include "NAF.h"

NAF::NAF(Environment*const _env, Settings & settings) :
Learner_offPolicy(_env, settings)
{
  F.push_back(new Approximator("value", settings, input, data));
  Builder build_pol = F[0]->buildFromSettings(settings, aInfo.dim);
  F[0]->initializeNetwork(build_pol);
  test();
}

void NAF::select(const Agent& agent)
{
  const Real annealedVar = greedyEps + (bTrain ? annealingFactor() : 0);
  const int thrID= omp_get_thread_num();
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);

  if( agent.Status != 2 )
  {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    vector<Real> output = F[0]->forward_agent<CUR>(traj, agent, thrID);
    const Quadratic_advantage advantage = prepare_advantage(output);
    vector<Real> pol = advantage.getMean();
    pol.resize(policyVecDim, annealedVar);
    const auto act = Gaussian_policy::sample(&generators[thrID], pol);
    agent.a->set(aInfo.getScaled(act));
    data->add_action(agent, pol);
  } else
    data->terminate_seq(agent);
}

void NAF::Train_BPTT(const Uint seq, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  const Uint ndata = traj->tuples.size();
  F[0]->prepare_seq(traj, thrID);
  if(thrID==1) profiler->stop_start("FWD");

  for (Uint k=0; k<ndata-1; k++) { //state in k=[0:N-2]
    const bool terminal = k+2==ndata && traj->ended;
    const vector<Real> output = F[0]->forward<CUR>(traj, k, thrID);
    const Real Vsold = output[net_indices[0]];
    const vector<Real> act = aInfo.getInvScaled(traj->tuples[k]->a);
    const Quadratic_advantage adv_sold = prepare_advantage(output);
    const Real Qsold = Vsold + adv_sold.computeAdvantage(act);

    Real Vsnew = traj->tuples[k+1]->r;
    if ( not terminal ) {
      const vector<Real> target = F[0]->forward<TGT>(traj, k+1, thrID);
      Vsnew += gamma*target[net_indices[0]];
    }
    const Real error = Vsnew - Qsold;
    vector<Real> gradient(F[0]->nOutputs());
    gradient[net_indices[0]] = error;
    adv_sold.grad(act, error, gradient);

    traj->SquaredError[k] = error*error;
    Vstats[thrID].dumpStats(Qsold, error);
    F[0]->backward(gradient, k, thrID);
  }

  if(thrID==1)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
  if(thrID==1)  profiler->stop_start("SLP");
}

void NAF::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  Sequence* const traj = data->Set[seq];
  F[0]->prepare_one(traj, samp, thrID);
  if(thrID==1) profiler->stop_start("FWD");

  const vector<Real> output = F[0]->forward<CUR>(traj, samp, thrID);
  const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  const Real Vsold = output[net_indices[0]];
  //unbounded action:
  const vector<Real> act = aInfo.getInvScaled(traj->tuples[samp]->a);
  const Quadratic_advantage adv_sold = prepare_advantage(output);
  const Real Qsold = Vsold + adv_sold.computeAdvantage(act);

  Real Vsnew = traj->tuples[samp+1]->r;
  if (not terminal) {
    const vector<Real> target = F[0]->forward<TGT>(traj, samp+1, thrID);
    Vsnew += gamma*target[net_indices[0]];
  }
  const Real error = Vsnew - Qsold;
  vector<Real> gradient(F[0]->nOutputs());
  gradient[net_indices[0]] = error;
  adv_sold.grad(act, error, gradient);

  traj->SquaredError[samp] = error*error;
  Vstats[thrID].dumpStats(Qsold, error);
  if(thrID==1)  profiler->stop_start("BCK");
  F[0]->backward(gradient, samp, thrID);
  F[0]->gradient(thrID);
  if(thrID==1)  profiler->stop_start("SLP");
}
