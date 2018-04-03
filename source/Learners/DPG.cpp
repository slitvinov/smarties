/*
 *  NFQ.cpp
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#include "../StateAction.h"
#include "../Math/Utils.h"
#include "../Math/Gaussian_policy.h"
#include "../Network/Builder.h"
#include "DPG.h"

DPG::DPG(Environment*const _env, Settings& _set) : Learner_offPolicy(_env, _set)
{
  _set.splitLayers = 0;
  #if 1
    if(input->net not_eq nullptr) {
      delete input->opt; input->opt = nullptr;
      delete input->net; input->net = nullptr;
    }
    Builder input_build(_set);
    bool bInputNet = false;
    input_build.addInput( input->nOutputs() );
    bInputNet = bInputNet || env->predefinedNetwork(input_build);
    bInputNet = bInputNet || predefinedNetwork(input_build, _set);
    if(bInputNet) {
      Network* net = input_build.build();
      input->initializeNetwork(net, input_build.opt);
    }
  #endif

  F.push_back(new Approximator("policy", _set, input, data));
  relay = new Aggregator(_set, data, nA, F[0]);
  F.push_back(new Approximator("value", _set, input, data, relay));
  Builder build_pol = F[0]->buildFromSettings(_set, nA);

  #if 1
    Builder build_val = F[1]->buildFromSettings(_set, 1 );
  #else
    Builder build_val(_set);
    build_val.stackSimple(input->nOutputs(), {0});
    build_val.addInput(relay->nOutputs()); // add actions
    build_val.addLayer(1, "Linear", true); // output
  #endif
  F[0]->initializeNetwork(build_pol, 10);
  F[0]->blockInpGrad = true;
  _set.learnrate *= 10; // DPG wants critic faster than actor
  _set.nnLambda = 1e-2; // also wants 1e-2 L2 penl coef
  _set.nnFunc = "LRelu"; // works best with rectifiers
  F[1]->initializeNetwork(build_val, 10);
  printf("DPG\n");
}

void DPG::select(const Agent& agent)
{
  const int thrID= omp_get_thread_num();
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);
  std::normal_distribution<Real> dist(0, 1);
  if( agent.Status < TERM_COMM ) { // not last of a sequence
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    Rvec pol = F[0]->forward_agent(traj, agent, thrID);
    Rvec act = pol;
    for(Uint i=0; i<nA; i++) {
      OrUhState[agent.ID][i] *= .85;
      pol[i] += OrUhState[agent.ID][i]; //is mean if we model pol as gaussian
      OrUhState[agent.ID][i] += greedyEps*dist(generators[thrID]);
      act[i] += OrUhState[agent.ID][i];
      pol.push_back(greedyEps); //is stdev if model pol as gaussian
    }
    agent.act(aInfo.getScaled(act));
    data->add_action(agent, pol);
  } else {
    OrUhState[agent.ID] = Rvec(nA, 0);
    data->terminate_seq(agent);
  }
}

void DPG::Train_BPTT(const Uint seq, const Uint thrID) const
{
  if(thrID==1) profiler->stop_start("FWD");
  Sequence* const traj = data->Set[seq];
  F[0]->prepare_seq(traj, thrID);
  F[1]->prepare_seq(traj, thrID);

  for (Uint k=0; k<traj->tuples.size()-1; k++)
  {
    const Rvec pol_curr = F[0]->forward<CUR>(traj, k, thrID);
    relay->prepare(ACT, thrID); // tell relay between two nets to pass actions
    const Rvec q_curr = F[1]->forward<CUR>(traj, k, thrID);

    relay->prepare(NET, thrID); // tell relay to pass policy (output of F[0])
    { //code to compute policy grad:
      //F[1] to compute v_curr (relay set NET, therefore input is policy)
      //with TGT weights on work memory of the TGT net.
      //if k>0, this was already computed in previous step of value grad
      const Rvec v_curr = F[1]->forward<TGT>(traj, k, thrID);

      //to use current weights (so, policy grad wrt to current value net)
      //we need to ask using CUR weights, overwriting the TGT work memory
      //because we will need the CUR work memory unpolluted for backprop:
      //const Rvec v_curr = F[1]->forward<CUR,TGT,1>(traj, k+1, thrID);

      const Rvec polGr = F[1]->relay_backprop({1}, k, thrID);
      F[0]->backward(polGr, k, thrID);
    }
    { //code to compute value grad:
      const bool terminal = k+2 == traj->tuples.size() && traj->ended;
      Real target = traj->tuples[k+1]->r;
      if (not terminal) {
        //unnecessary function call, just to make it transaprent to reader:
        /*const Rvec pol_next = */ F[0]->forward<CUR>(traj, k+1, thrID);
        // relay NET is still in effect, this will take pol_next as input:
        const Rvec v_next = F[1]->forward<TGT>(traj, k+1, thrID);
        target += gamma * v_next[0];
      }

      const Rvec grad_val = {(target-q_curr[0])};
      traj->SquaredError[k] = grad_val[0]*grad_val[0];
      Vstats[thrID].dumpStats(q_curr[0], grad_val[0]);
      F[1]->backward(grad_val, k, thrID);
    }
  }


  if(thrID==1)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
}

void DPG::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  if(thrID==1) profiler->stop_start("FWD");
  Sequence* const traj = data->Set[seq];
  const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  F[0]->prepare_one(traj, samp, thrID);
  F[1]->prepare_one(traj, samp, thrID);

  relay->prepare(ACT, thrID); // tell relay between two nets to pass actions
  const Rvec pol_curr = F[0]->forward(traj, samp, thrID);
  const Rvec q_curr = F[1]->forward(traj, samp, thrID);

  relay->prepare(NET, thrID); // tell relay to pass policy (output of F[0])
  { //code to compute policy grad:
    //const Rvec v_curr = F[1]->forward<TGT>(traj, samp, thrID);
    const Rvec v_curr = F[1]->forward<CUR, TGT>(traj, samp, thrID);
    const Rvec polGr = F[1]->relay_backprop({1}, samp, thrID);
    //cout <<"Inp grad: "<< print(polGr) << endl; fflush(0);
    F[0]->backward(polGr, samp, thrID);
  }

  { //code to compute value grad:
    Real target = data->scaledReward(traj,samp+1);
    if (not terminal) {
      const Rvec pol_next = F[0]->forward(traj, samp+1, thrID);
      const Rvec v_next = F[1]->forward<TGT>(traj, samp+1, thrID);
      target += gamma * v_next[0];
    }

    const Rvec grad_val = {(target-q_curr[0])};
    traj->SquaredError[samp] = grad_val[0]*grad_val[0];
    Vstats[thrID].dumpStats(q_curr[0], grad_val[0]);
    F[1]->backward(grad_val, samp, thrID);
  }
  if(thrID==1)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
}
