//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#include "../StateAction.h"
#include "../Network/Builder.h"
#include "../Network/Aggregator.h"
#include "../Math/Gaussian_policy.h"
#include "DPG.h"
//#define DKL_filter

static inline Gaussian_policy prepare_policy(const Rvec&out,
  const ActionInfo*const aI, const Tuple*const t = nullptr) {
  Gaussian_policy pol({0, aI->dim}, aI, out);
  if(t not_eq nullptr) pol.prepare(t->a, t->mu);
  return pol;
}

void DPG::TrainBySequences(const Uint seq, const Uint wID, const Uint bID,
  const Uint thrID) const
{
  die("");
}

void DPG::Train(const Uint seq, const Uint t, const Uint wID,
  const Uint bID, const Uint thrID) const
{
  if(thrID==0) profiler->stop_start("FWD");
  Sequence* const traj = data->get(seq);
  F[0]->prepare_one(traj, t, thrID, wID);
  F[1]->prepare_one(traj, t, thrID, wID);

  const Rvec polVec = F[0]->forward_cur(t, thrID);
  const Gaussian_policy POL = prepare_policy(polVec, &aInfo, traj->tuples[t]);
  const Real DKL = POL.sampKLdiv, rho = POL.sampImpWeight;
  //if(!thrID) cout<<"tpol "<<print(polVec)<<" act: "<<print(POL.sampAct)<<endl;
  #ifdef REFER_RULE1
    const bool isOff = traj->isFarPolicy(t, rho, CmaxRet,CinvRet);
  #else
    const bool isOff = false;
  #endif

  relay->prepare(traj, thrID, ACT);
  const Rvec q_curr = F[1]->forward_cur(t, thrID); // inp here is {s,a}

  relay->prepare(traj, thrID, NET); // relay to pass policy (output of F[0])
  const Rvec v_curr = F[1]->forward_cur<TGT>(t, thrID); //here is {s,pi}
  const Rvec detPolG = isOff? Rvec(nA,0) : F[1]->relay_backprop({1}, t, thrID);
  //if(!thrID) cout << "G "<<print(detPolG) << endl;

  Real target = data->scaledReward(traj, t+1);
  if (not traj->isTerminal(t+1) && not isOff) {
    const Rvec pol_next = F[0]->forward_tgt(t+1, thrID);
    relay->prepare(traj, thrID, NET); // relay to pass policy (output of F[0])
    //if(!thrID) cout << "nterm pol "<<print(pol_next) << endl;
    const Rvec v_next = F[1]->forward_tgt(t+1, thrID);//here is {s,pi}_+1
    target += gamma * v_next[0];
  }

  //code to compute policy grad:
  Rvec polG(2*nA, 0);
  for (Uint i=0; i<nA; i++) polG[i] = isOff? 0 : detPolG[i];
  for (Uint i=0; i<nA; i++) polG[i+nA] = explNoise - POL.stdev[i];
  // this is an experimental change to update stdev using policy gradient
  // not fully analyzed therefore should be turned off by default
  //Real a_curr = target - v_curr[0];
  //if(a_curr > 0 && POL.sampImpWeight >  2) a_curr = 0;
  //if(a_curr < 0 && POL.sampImpWeight < .5) a_curr = 0;
  //Rvec sPG = POL.policy_grad(POL.sampAct, POL.sampImpWeight*a_curr);

  const Rvec penG = POL.div_kl_grad(traj->tuples[t]->mu, -1);
  // if beta=1 (which is inevitable for CmaxPol=0) this will be equal to polG
  #ifdef REFER_RULE3
    const Rvec mixG = weightSum2Grads(polG, penG, beta);
  #else
    const Rvec mixG = polG;
  #endif
  Rvec finalG(F[0]->nOutputs(), 0);
  POL.finalize_grad(mixG, finalG);
  //#pragma omp critical //"O:"<<print(polVec)<<
  //if(!thrID) cout<<"G:"<<print(polG)<<" D:"<<print(penG)<<endl;
  F[0]->backward(finalG, t, thrID);


  //code to compute value grad:
  const Rvec grad_val = {isOff ? 0 : (target-q_curr[0])};
  F[1]->backward(grad_val, t, thrID);

  //bookkeeping:
  trainInfo->log(q_curr[0], grad_val[0], polG, penG, {beta,rho}, thrID);
  traj->setMseDklImpw(t, grad_val[0]*grad_val[0], DKL, rho, CmaxRet, CinvRet);
  if(thrID==0)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
}

void DPG::select(Agent& agent)
{
  Sequence* const traj = data_get->get(agent.ID);
  data_get->add_state(agent);
  if( agent.Status < TERM_COMM ) { // not last of a sequence
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    F[0]->prepare_agent(traj, agent);
    Rvec pol = F[0]->forward_agent(agent);
    Gaussian_policy policy = prepare_policy(pol, &aInfo);
    Rvec MU = policy.getVector();
    Rvec act = policy.finalize(explNoise>0, &generators[nThreads+agent.ID], MU);
    if(OrUhDecay>0)
      act = policy.updateOrUhState(OrUhState[agent.ID], MU, OrUhDecay);
    agent.act(act);
    data_get->add_action(agent, MU);
    //if(nStep)cout << print(MU) << " " << print(act) << endl;
  } else {
    OrUhState[agent.ID] = Rvec(nA, 0);
    data_get->terminate_seq(agent);
  }
}

DPG::DPG(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set)
{
  _set.splitLayers = 0;
  #if 0
    createSharedEncoder();
  #endif

  F.push_back(new Approximator("policy", _set, input, data));
  Builder build_pol = F[0]->buildFromSettings(_set, nA);
  #ifdef EXTRACT_COVAR
    const Real stdParam = noiseMap_inverse(explNoise*explNoise);
  #else
    const Real stdParam = noiseMap_inverse(explNoise);
  #endif
  //F[0]->blockInpGrad = true; // this line must happen b4 initialize
  build_pol.addParamLayer(nA, "Linear", stdParam);
  F[0]->initializeNetwork(build_pol);

  relay = new Aggregator(_set, data, nA, F[0]);
  F.push_back(new Approximator("critic", _set, input, data, relay));

  _set.nnLambda = 1e-4; // also wants L2 penl coef
  _set.learnrate *= 10; // DPG wants critic faster than actor
  _set.nnOutputFunc = "Linear"; // critic must be linear
  // we want initial Q to be approx equal to 0 everywhere.
  // if LRelu we need to make initialization multiplier smaller:
  Builder build_val = F[1]->buildFromSettings(_set, 1 );
  F[1]->initializeNetwork(build_val);
  printf("DPG\n");

  trainInfo = new TrainData("DPG", _set, 1, "| beta | avgW ", 2);
}
