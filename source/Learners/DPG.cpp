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

DPG::DPG(Environment*const _env, Settings& sett) : Learner_offPolicy(_env, sett)
{
  F.push_back(new Approximator("policy", sett, input, data));
  relay = new Aggregator(sett, data, _env->aI.dim, F[0]);
  F.push_back(new Approximator("value", sett, input, data, relay));
  Builder build_pol = F[0]->buildFromSettings(sett, _env->aI.dim);
  Builder build_val = F[1]->buildFromSettings(sett, 1 );
  F[0]->initializeNetwork(build_pol);
  F[1]->initializeNetwork(build_val);
}

void DPG::select(const Agent& agent)
{
  const Real annealedVar = greedyEps + (bTrain ? annealingFactor() : 0);
  const int thrID= omp_get_thread_num();
  Sequence* const traj = data->inProgress[agent.ID];
  data->add_state(agent);

  if( agent.Status != 2 ) {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    vector<Real> pol = F[0]->forward_agent<CUR>(traj, agent, thrID);
    pol.resize(policyVecDim, annealedVar);
    const auto act = Gaussian_policy::sample(&generators[thrID], pol);
    agent.act(aInfo.getScaled(act));
    data->add_action(agent, pol);
  } else
    data->terminate_seq(agent);
}

void DPG::Train_BPTT(const Uint seq, const Uint thrID) const
{
  if(thrID==1) profiler->stop_start("FWD");
  Sequence* const traj = data->Set[seq];
  F[0]->prepare_seq(traj, thrID);
  F[1]->prepare_seq(traj, thrID);

  for (Uint k=0; k<traj->tuples.size()-1; k++)
  {
    const vector<Real> pol_curr = F[0]->forward<CUR>(traj, k, thrID);
    relay->prepare(ACT, thrID); // tell relay between two nets to pass actions
    const vector<Real> q_curr = F[1]->forward<CUR>(traj, k, thrID);

    relay->prepare(NET, thrID); // tell relay to pass policy (output of F[0])
    { //code to compute policy grad:
      //F[1] to compute v_curr (relay set NET, therefore input is policy)
      //with TGT weights on work memory of the TGT net.
      //if k>0, this was already computed in previous step of value grad
      const vector<Real> v_curr = F[1]->forward<TGT>(traj, k, thrID);

      //to use current weights (so, policy grad wrt to current value net)
      //we need to ask using CUR weights, overwriting the TGT work memory
      //because we will need the CUR work memory unpolluted for backprop:
      //const vector<Real> v_curr = F[1]->forward<CUR,TGT,1>(traj, k+1, thrID);

      const vector<Real> polGr = F[1]->relay_backprop<TGT>({1}, k, thrID);
      F[0]->backward(polGr, k, thrID);
    }
    { //code to compute value grad:
      const bool terminal = k+2 == traj->tuples.size() && traj->ended;
      Real target = traj->tuples[k+1]->r;
      if (not terminal) {
        //unnecessary function call, just to make it transaprent to reader:
        /*const vector<Real> pol_next = */ F[0]->forward<CUR>(traj, k+1, thrID);
        // relay NET is still in effect, this will take pol_next as input:
        const vector<Real> v_next = F[1]->forward<TGT>(traj, k+1, thrID);
        target += gamma * v_next[0];
      }

      const vector<Real> grad_val = {target - q_curr[0]};
      traj->SquaredError[k] = grad_val[0]*grad_val[0];
      Vstats[thrID].dumpStats(q_curr[0], grad_val[0]);
      F[1]->backward(grad_val, k, thrID);
    }
  }


  if(thrID==1)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
  if(thrID==1)  profiler->stop_start("SLP");
}

void DPG::Train(const Uint seq, const Uint samp, const Uint thrID) const
{
  if(thrID==1) profiler->stop_start("FWD");
  Sequence* const traj = data->Set[seq];
  const bool terminal = samp+2 == traj->tuples.size() && traj->ended;
  F[0]->prepare_one(traj, samp, thrID);
  F[1]->prepare_one(traj, samp, thrID);

  relay->prepare(ACT, thrID); // tell relay between two nets to pass actions
  const vector<Real> pol_curr = F[0]->forward<CUR>(traj, samp, thrID);
  const vector<Real> q_curr = F[1]->forward<CUR>(traj, samp, thrID);

  relay->prepare(NET, thrID); // tell relay to pass policy (output of F[0])
  { //code to compute policy grad:
    //const vector<Real> v_curr = F[1]->forward<TGT>(traj, samp, thrID);
    const vector<Real> v_curr = F[1]->forward<CUR, TGT>(traj, samp, thrID);
    const vector<Real> polGr = F[1]->relay_backprop<TGT>({1}, samp, thrID);

    F[0]->backward(polGr, samp, thrID);
  }

  { //code to compute value grad:
    Real target = traj->tuples[samp+1]->r;
    if (not terminal) {
      const vector<Real> pol_next = F[0]->forward<CUR>(traj, samp+1, thrID);
      const vector<Real> v_next = F[1]->forward<TGT>(traj, samp+1, thrID);
      target += gamma * v_next[0];
    }

    const vector<Real> grad_val = {target - q_curr[0]};
    traj->SquaredError[samp] = grad_val[0]*grad_val[0];
    Vstats[thrID].dumpStats(q_curr[0], grad_val[0]);
    F[1]->backward(grad_val, samp, thrID);
  }
  if(thrID==1)  profiler->stop_start("BCK");
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
  if(thrID==1)  profiler->stop_start("SLP");
}
