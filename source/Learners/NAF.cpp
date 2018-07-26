//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../StateAction.h"
#include "../Network/Builder.h"
#include "NAF.h"

NAF::NAF(Environment*const _env, Settings & _set) :
Learner_offPolicy(_env, _set)
{
  trainInfo = new TrainData("NAF", _set, 0, "| beta | avgW ", 2);
  F.push_back(new Approximator("value", _set, input, data));
  const Uint nOutp = 1 +aInfo.dim +Quadratic_advantage::compute_nL(&aInfo);
  Builder build_pol = F[0]->buildFromSettings(_set, nOutp);
  F[0]->initializeNetwork(build_pol, 0);
  test();
}

void NAF::select(Agent& agent)
{
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);

  if( agent.Status < TERM_COMM ) // not last of a sequence
  {
    F[0]->prepare_agent(traj, agent);
    //Compute policy and value on most recent element of the sequence.
    const Rvec output = F[0]->forward_agent(traj, agent);
    //cout << print(output) << endl;
    const Quadratic_advantage advantage = prepare_advantage(output);
    Rvec pol = advantage.getMean();
    pol.resize(policyVecDim, explNoise);
    assert(pol.size() == 2 * nA);
    Gaussian_policy policy({0, nA}, &aInfo, pol);
    const Rvec MU = policy.getVector();
    //cout << print(MU) << endl;
    Rvec act = policy.finalize(explNoise>0, &generators[nThreads+agent.ID], MU);
    if(OrUhDecay>0)
      act = policy.updateOrUhState(OrUhState[agent.ID], MU, OrUhDecay);

    agent.act(act);
    data->add_action(agent, MU);
  } else {
    OrUhState[agent.ID] = Rvec(nA, 0);
    data->terminate_seq(agent);
  }
}

void NAF::TrainBySequences(const Uint seq, const Uint thrID) const
{
  die("");
  Sequence* const traj = data->Set[seq];
  const Uint ndata = traj->tuples.size();
  F[0]->prepare_seq(traj, thrID);
  if(thrID==0) profiler->stop_start("FWD");

  for (Uint k=0; k<ndata-1; k++) { //state in k=[0:N-2]
    const bool terminal = k+2==ndata && traj->ended;
    const Rvec output = F[0]->forward<CUR>(traj, k, thrID);
    const Real Vsold = output[net_indices[0]];
    const Rvec act = aInfo.getInvScaled(traj->tuples[k]->a);
    const Quadratic_advantage adv_sold = prepare_advantage(output);
    const Real Qsold = Vsold + adv_sold.computeAdvantage(act);

    Real Vsnew = traj->tuples[k+1]->r;
    if ( not terminal ) {
      const Rvec target = F[0]->forward<TGT>(traj, k+1, thrID);
      Vsnew += gamma*target[net_indices[0]];
    }
    const Real error = Vsnew - Qsold;
    Rvec gradient(F[0]->nOutputs());
    gradient[net_indices[0]] = error;
    adv_sold.grad(act, error, gradient);

    traj->setMseDklImpw(k, error*error, 0, 1);
    trainInfo->log(Qsold, error, thrID);
    F[0]->backward(gradient, k, thrID);
  }

  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
}

void NAF::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  if(thrID==0) profiler->stop_start("FWD");

  Sequence* const traj = data->Set[seq];
  F[0]->prepare_one(traj, samp, thrID);

  const Rvec output = F[0]->forward(traj, samp, thrID);
  //const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  const Real Vsold = output[net_indices[0]];

  // prepare advantage and policy
  const Quadratic_advantage adv_sold = prepare_advantage(output);
  Rvec polvec = adv_sold.getMean();       assert(polvec.size() == nA);
  polvec.resize(policyVecDim, explNoise); assert(polvec.size() == 2 * nA);
  Gaussian_policy policy({0, nA}, &aInfo, polvec);
  policy.prepare(traj->tuples[samp]->a, traj->tuples[samp]->mu);
  const Real rho = policy.sampImpWeight, dkl = policy.sampKLdiv;
  //cout << rho << " " << dkl << " " << CmaxRet << endl;

  const Real Qsold = Vsold + adv_sold.computeAdvantage(policy.sampAct);
  const bool isOff = traj->isFarPolicy(samp, rho, CmaxRet);

  Real Vsnew = data->scaledReward(traj, samp+1);
  if (not traj->isTerminal(samp+1) && not isOff) {
    const Rvec target = F[0]->forward<TGT>(traj, samp+1, thrID);
    Vsnew += gamma*target[net_indices[0]];
  }
  const Real error = isOff? 0 : Vsnew - Qsold;
  Rvec grad(F[0]->nOutputs());
  grad[net_indices[0]] = error;
  adv_sold.grad(policy.sampAct, error, grad);
  if(CmaxRet>1 && beta<1) { // then ReFER
    const Rvec penG = policy.div_kl_grad(traj->tuples[samp]->mu, -1);
    for(Uint i=0; i<nA; i++)
      grad[net_indices[2]+i] = beta*grad[net_indices[2]+i] + (1-beta)*penG[i];
  }

  trainInfo->log(Qsold, error, {beta,rho}, thrID);
  traj->setMseDklImpw(samp, error*error, dkl, rho);
  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->backward(grad, samp, thrID);
  F[0]->gradient(thrID);
}
