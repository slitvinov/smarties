/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

//#define dumpExtra
#ifndef DACER_SKIP
#define DACER_SKIP 1
#endif

#define DACER_BACKWARD
//#ifndef DACER_FORWARD
#define DACER_FORWARD 0
//#endif

template<typename Policy_t, typename Action_t>
class DACER : public Learner_offPolicy
{
 protected:
  // continuous actions: dimensionality of action vectors
  // discrete actions: number of options
  const Uint nA = Policy_t::compute_nA(&aInfo);

  // tgtFrac_param: target fraction of off-pol samples
  // alpha: weight of value-update relative to policy update. 1 means equal
  const Real alpha=1;
  Real CmaxRet = 1 + CmaxPol;

  // indices identifying number and starting position of the different output // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const vector<Uint> net_outputs, net_indices, pol_start;
  const Uint VsID = net_indices[0];

  // used in case of temporally correlated noise
  vector<Rvec> OrUhState = vector<Rvec>( nAgents, Rvec(nA, 0) );

  inline Policy_t prepare_policy(const Rvec& out,
    const Tuple*const t = nullptr) const {
    Policy_t pol(pol_start, &aInfo, out);
    // pol.prepare computes various quanties that depend on behavioral policy mu
    // (such as importance weight) and stores both mu and the non-scaled action

    //policy saves pol.sampAct, which is unscaled action
    //eg. if action bounds act in [-1 1]; learning is with sampAct in (-inf inf)
    // when facing out of the learner we output act = tanh(sampAct)
    // TODO semi-bounded action spaces! eg. [0 inf): act = softplus(sampAct)
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }

  void prepareData()
  {
    Learner_offPolicy::prepareData();
    if(updatePrepared && nStep == 0) {
      #pragma omp parallel for schedule(dynamic)
      for(Uint i = 0; i < data->Set.size(); i++) {
        Sequence* const traj = data->Set[i];
        const int N = traj->ndata(); traj->setRetrace(N, 0);
        for(Uint j=N; j>0; j--) updateVret(traj, j-1, traj->state_vals[j-1], 1);
      }
    }
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override
  {
    die("");
  }

  void Train(const Uint seq, const Uint samp, const Uint thrID) const override
  {
    Sequence* const traj = data->Set[seq];
    assert(samp+1 < traj->tuples.size());

    if(thrID==1) profiler->stop_start("FWD");

    F[0]->prepare_one(traj, samp, thrID); // prepare thread workspace
    const Rvec out_cur = F[0]->forward(traj, samp, thrID); // network compute

    if( traj->isTruncated(samp+1) ) {
      const Rvec nxt = F[0]->forward(traj, samp+1, thrID);
      traj->setStateValue(samp+1, nxt[VsID]);
    }

    const Policy_t pol = prepare_policy(out_cur, traj->tuples[samp]);
    // check whether importance weight is in 1/Cmax < c < Cmax
    const bool isOff = traj->isFarPolicy(samp, pol.sampImpWeight, CmaxRet);

    if(thrID==1)  profiler->stop_start("CMP");
    Rvec grad;

    #if   DACER_SKIP == 1
      if(isOff) grad = offPolGrad(traj, samp, out_cur, pol, thrID);
      else
    #endif
        grad = compute(traj, samp, out_cur, pol, thrID);

    if(thrID==1)  profiler->stop_start("BCK");
    F[0]->backward(grad, samp, thrID); // place gradient onto output layer
    F[0]->gradient(thrID);  // backprop
  }

  inline Rvec compute(Sequence*const S, const Uint t, const Rvec& outVec,
    const Policy_t& pol_cur, const Uint thrID) const
  {
    const Real rNext = data->scaledReward(S, t+1), V_cur = outVec[VsID];
    const Real A_RET = rNext +gamma*(S->Q_RET[t+1]+S->state_vals[t+1]) - V_cur;
    //prepare Q with off policy corrections for next step:
    const Real dAdv = updateVret(S, t, V_cur, pol_cur);
    const Real rho_cur = pol_cur.sampImpWeight, Ver = S->Q_RET[t];

    const Rvec policyG = pol_cur.policy_grad(pol_cur.sampAct, A_RET*rho_cur);
    const Rvec penalG  = pol_cur.div_kl_grad(S->tuples[t]->mu, -1);
    const Rvec finalG  = weightSum2Grads(policyG, penalG, beta);

    Rvec gradient(F[0]->nOutputs(), 0);
    gradient[VsID] = beta*alpha *Ver;
    pol_cur.finalize_grad(finalG, gradient);
    //bookkeeping:
    trainInfo->log(V_cur, Ver, policyG, penalG, {beta, dAdv, rho_cur}, thrID);
    S->setMseDklImpw(t, S->Q_RET[t]*S->Q_RET[t], pol_cur.sampKLdiv, rho_cur);
    return gradient;
  }

  inline Rvec offPolGrad(Sequence*const S, const Uint t, const Rvec output,
    const Policy_t& pol, const Uint thrID) const {
    updateVret(S, t, output[VsID], pol);
    S->setMseDklImpw(t,std::pow(S->Q_RET[t],2),pol.sampKLdiv,pol.sampImpWeight);
    // prepare penalization gradient:
    Rvec gradient(F[0]->nOutputs(), 0);
    const Rvec pg = pol.div_kl_grad(S->tuples[t]->mu, beta-1);
    pol.finalize_grad(pg, gradient);
    return gradient;
  }

  inline Real updateVret(Sequence*const S, const Uint t, const Real V,
    const Policy_t& pol) const {
    return updateVret(S, t, V, pol.sampImpWeight);
  }

  inline Real updateVret(Sequence*const S, const Uint t, const Real V,
    const Real rho) const {
    assert(rho >= 0);
    const Real rNext = data->scaledReward(S, t+1), oldVret = S->Q_RET[t];
    const Real vNext = S->state_vals[t+1], V_RET = S->Q_RET[t+1];
    const Real delta = std::min((Real)1, rho) * (rNext +gamma*vNext -V);
    //const Real trace = gamma *.95 *std::pow(rho,retraceTrickPow) *V_RET;
    const Real trace = gamma *std::min((Real)1, rho) *V_RET;
    S->setStateValue(t, V ); S->setRetrace(t, delta + trace);
    return std::fabs(S->Q_RET[t] - oldVret);
  }

 public:
  DACER(Environment*const _env, Settings& _set, vector<Uint> net_outs,
   vector<Uint> pol_inds): Learner_offPolicy(_env,_set), net_outputs(net_outs),
   net_indices(count_indices(net_outs)), pol_start(pol_inds)
  {
    printf("DACER starts: v:%u pol:%s\n", VsID, print(pol_start).c_str());
    trainInfo = new TrainData("v-racer", _set, 1, "| beta | dAdv | avgW ", 3);
  }
  ~DACER() { }

  void select(const Agent& agent) override
  {
    const int thrID= omp_get_thread_num();
    Sequence* const traj = data->inProgress[agent.ID];
    data->add_state(agent);

    if( agent.Status < TERM_COMM ) // not last of a sequence
    {
      //Compute policy and value on most recent element of the sequence. If RNN
      // recurrent connection from last call from same agent will be reused
      Rvec output = F[0]->forward_agent(traj, agent, thrID);
      Policy_t pol = prepare_policy(output);
      Rvec mu = pol.getVector(); // vector-form current policy for storage

      // if explNoise is 0, we just act according to policy
      // since explNoise is initial value of diagonal std vectors
      // this should only be used for evaluating a learned policy
      Action_t act = pol.finalize(explNoise>0, &generators[thrID], mu);

      #if 0 // add and update temporally correlated noise
        act = pol.updateOrUhState(OrUhState[agent.ID], mu, act, iter());
      #endif

      traj->state_vals.push_back(output[VsID]);
      agent.a->set(act);
      data->add_action(agent, mu);

      #ifndef NDEBUG
        //Policy_t dbg = prepare_policy(output);
        //dbg.prepare(traj->tuples.back()->a, traj->tuples.back()->mu);
        //const double err = fabs(dbg.sampImpWeight-1);
        //if(err>1e-10) _die("Imp W err %20.20e", err);
      #endif
    }
    else
    {
      if( agent.Status == TRNC_COMM ) {
        Rvec output = F[0]->forward_agent(traj, agent, thrID);
        traj->state_vals.push_back(output[VsID]);
      } else
        traj->state_vals.push_back(0); //value of terminal state is 0

      writeOnPolRetrace(traj); // compute initial Qret for whole trajectory
      OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
      data->terminate_seq(agent);
    }
  }

  void writeOnPolRetrace(Sequence*const seq) const
  {
    assert(seq->tuples.size() == seq->state_vals.size());
    assert(seq->Q_RET.size() == 0);
    const Uint N = seq->tuples.size();
    //within Retrace, we use the state_vals vector to write the Q retrace values
    seq->Q_RET.resize(N, 0);
    //TODO extend for non-terminal trajectories: one more v_state predict

    seq->Q_RET[N-1] = 0; //both if truncated or not, delta is zero

    //update all q_ret before terminal step
    for (Uint i=N-1; i>0; i--) updateVret(seq, i-1, seq->state_vals[i-1], 1);
  }

  void prepareGradient()
  {
    #ifdef RACER_BACKWARD
    if(updateComplete) {
      profiler->stop_start("QRET");
      #pragma omp parallel for schedule(dynamic)
      for(Uint i = 0; i < data->Set.size(); i++)
        for(int j = data->Set[i]->just_sampled; j>=0; j--)
          updateVret(data->Set[i], j, data->Set[i]->state_vals[j], data->Set[i]->offPolicImpW[j]);
    }
    #endif

    Learner_offPolicy::prepareGradientReFER();
  }
};
