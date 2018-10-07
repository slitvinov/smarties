//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "RETPG.h"
#include "../Network/Builder.h"
#include "../Network/Aggregator.h"

void RETPG::TrainBySequences(const Uint seq, const Uint wID, const Uint bID,
  const Uint thrID) const {
  die(" ");
}

void RETPG::Train(const Uint seq, const Uint t, const Uint wID,
  const Uint bID, const Uint thrID) const
{
  Sequence* const traj = data->get(seq);
  assert(t+1 < traj->tuples.size());

  if(thrID==0) profiler->stop_start("FWD");

  F[0]->prepare_one(traj, t, thrID, wID);
  F[1]->prepare_one(traj, t, thrID, wID);

  const Rvec polVec = F[0]->forward_cur(t, thrID); // network compute
  relay->prepare(traj, thrID, ACT);
  const Rvec q_curr = F[1]->forward_cur(t, thrID); // inp here is {s,a}

  const Gaussian_policy POL = prepare_policy(polVec, traj->tuples[t]);
  const Real DKL = POL.sampKLdiv, rho = POL.sampImpWeight;
  const bool isOff = traj->isFarPolicy(t, rho, CmaxRet);

  relay->prepare(traj, thrID, NET); //relay to pass policy (output of F[0])
  const Rvec v_curr = F[1]->forward_cur<TGT>(t, thrID); //here is {s,pi}
  const Rvec detPolG = isOff? Rvec(nA,0) : F[1]->relay_backprop({1}, t, thrID);

  if( traj->isTerminal(t+1) ) updateQret(traj, t+1, 0, 0, 0);
  else if( traj->isTruncated(t+1) ) {
    F[0]->forward(t+1, thrID);
    relay->prepare(traj, thrID, NET); // tell relay to pass policy (output of F[0])
    const Rvec v_next = F[1]->forward(t+1, thrID); //here is {s,pi}_+1
    traj->setStateValue(t+1, v_next[0]);
    updateQret(traj, t+1, 0, v_next[0], 0);
  }

  //code to compute policy grad:
  Rvec polG(2*nA, 0);
  for (Uint i=0; i<nA; i++) polG[i] = isOff? 0 : detPolG[i];
  // this keeps stdev at user's value, else NN penalization might cause drift:
  for (Uint i=0; i<nA; i++) polG[i+nA] = explNoise - POL.stdev[i];
  const Rvec penG = POL.div_kl_grad(traj->tuples[t]->mu, -1);
  // If beta=1 (which is inevitable for CmaxPol=0) this will be equal to polG:
  Rvec mixG = weightSum2Grads(polG, penG, beta);
  Rvec finalG = Rvec(F[0]->nOutputs(), 0);
  POL.finalize_grad(mixG, finalG);
  F[0]->backward(finalG, t, thrID);

  //code to compute value grad. Q_RET holds adv, sum with previous est of state
  // val: analogous to having target weights in original DPG
  const Real retTarget = traj->Q_RET[t] +traj->state_vals[t];
  const Rvec grad_val={isOff? 0: retTarget - q_curr[0]};
  F[1]->backward(grad_val, t, thrID);

  //bookkeeping:
  const Real dAdv = updateQret(traj, t, q_curr[0]-v_curr[0], v_curr[0], POL);
  trainInfo->log(q_curr[0], grad_val[0], polG, penG, {beta, dAdv, rho}, thrID);
  traj->setMseDklImpw(t, grad_val[0]*grad_val[0], DKL, rho);
  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
}

void RETPG::select(Agent& agent)
{
  Sequence* const traj = data_get->get(agent.ID);
  data_get->add_state(agent);
  F[0]->prepare_agent(traj, agent);
  F[1]->prepare_agent(traj, agent);

  if( agent.Status < TERM_COMM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    Rvec pol = F[0]->forward_agent(agent);
    Gaussian_policy policy = prepare_policy(pol);
    Rvec MU = policy.getVector();
    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy with bTrain=0
    Rvec act = policy.finalize(explNoise>0, &generators[nThreads+agent.ID], MU);
    if(OrUhDecay>0)
      act = policy.updateOrUhState(OrUhState[agent.ID], MU, OrUhDecay);
    agent.act(act);
    data_get->add_action(agent, MU);

    relay->prepare(traj, nThreads+agent.ID, ACT);
    const Rvec qval = F[1]->forward_agent(agent);
    relay->prepare(traj, nThreads+agent.ID, NET);
    const Rvec sval = F[1]->forward_agent(agent);
    traj->action_adv.push_back(qval[0]-sval[0]);
    traj->state_vals.push_back(sval[0]);
  }
  else
  {
    if( agent.Status == TRNC_COMM ) // not a terminal state
    { //then compute on policy state value
      F[0]->forward_agent(agent); // compute policy
      relay->prepare(traj, nThreads+agent.ID, NET); // pass it to q net
      const Rvec sval = F[1]->forward_agent(agent);
      traj->state_vals.push_back(sval[0]);
    } else
      traj->state_vals.push_back(0); //value of terminal state is 0
    //whether seq is truncated or terminated, act adv is undefined:
    traj->action_adv.push_back(0);

    // compute initial Qret for whole trajectory:
    assert(traj->tuples.size() == traj->action_adv.size());
    assert(traj->tuples.size() == traj->state_vals.size());
    assert(traj->Q_RET.size()  == 0);
    //within Retrace, we use the state_vals vector to write the Q retrace values
    traj->Q_RET.resize(traj->tuples.size(), 0);
    for(Uint i=traj->tuples.size()-1; i>0; i--) updateQretFront(traj, i);

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
}

void RETPG::prepareGradient()
{
  Learner_offPolicy::prepareGradient();

  if(updateToApply)
  {
    debugL("Update Retrace est. for episodes samples in prev. grad update");
    // placed here because this happens right after update is computed
    // this can happen before prune and before workers are joined
    profiler->stop_start("QRET");
    const Uint setSize = data->readNSeq();
    #pragma omp parallel for schedule(dynamic)
    for(Uint i=0; i<setSize; i++)
      for(int j=data->get(i)->just_sampled-1; j>0; j--)
        updateQretBack(data->get(i), j);
  }
}

void RETPG::initializeLearner()
{
  Learner_offPolicy::initializeLearner();

  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) Retrace values for all obss
  // seen before this point.
  debugL("Rescale Retrace est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics
  const Uint setSize = data->readNSeq();
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < setSize; i++)
  for(Uint j=data->get(i)->ndata(); j>0; j--) updateQretFront(data->get(i),j);
}

RETPG::RETPG(Environment*const _e, Settings& _s) : Learner_offPolicy(_e, _s)
{
  _s.splitLayers = 0;
  #if 0
    if(input->net not_eq nullptr) {
      delete input->opt; input->opt = nullptr;
      delete input->net; input->net = nullptr;
    }
    Builder input_build(_s);
    bool bInputNet = false;
    input_build.addInput( input->nOutputs() );
    bInputNet = bInputNet || env->predefinedNetwork(input_build);
    bInputNet = bInputNet || predefinedNetwork(input_build);
    if(bInputNet) {
      Network* net = input_build.build(true);
      input->initializeNetwork(net, input_build.opt);
    }
  #endif

  F.push_back(new Approximator("policy", _s, input, data));
  Builder build_pol = F[0]->buildFromSettings(_s, nA);
  const Real initParam = noiseMap_inverse(explNoise);
  //F[0]->blockInpGrad = true; // this line must happen b4 initialize
  build_pol.addParamLayer(nA, "Linear", initParam);
  F[0]->initializeNetwork(build_pol, 0);

  relay = new Aggregator(_s, data, nA, F[0]);
  F.push_back(new Approximator("critic", _s, input, data, relay));
  //relay->scaling = Rvec(nA, 1/explNoise);

  _s.nnLambda = 1e-4; // also wants L2 penl coef
  _s.learnrate *= 10; // DPG wants critic faster than actor
  _s.nnOutputFunc = "Linear"; // critic must be linear
  // we want initial Q to be approx equal to 0 everywhere.
  // if LRelu we need to make initialization multiplier smaller:
  Builder build_val = F[1]->buildFromSettings(_s, 1 );
  F[1]->initializeNetwork(build_val, 0);
  printf("DPG with Retrace-based Q network target\n");

  trainInfo = new TrainData("DPG", _s, 1, "| beta | dAdv | avgW ", 3);
}
