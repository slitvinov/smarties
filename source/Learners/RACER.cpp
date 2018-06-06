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
#ifndef RACER_SKIP
#define RACER_SKIP 1
#endif

#define RACER_BACKWARD
//#ifndef RACER_FORWARD
#define RACER_FORWARD 0
//#endif
#ifdef DKL_filter
#undef DKL_filter
#endif

template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_offPolicy
{
 protected:
  // continuous actions: dimensionality of action vectors
  // discrete actions: number of options
  const Uint nA = Policy_t::compute_nA(&aInfo);
  // number of parameters of advantage approximator
  const Uint nL = Advantage_t::compute_nL(&aInfo);

  // tgtFrac_param: target fraction of off-pol samples
  // alpha: weight of value-update relative to policy update. 1 means equal
  const Real tgtFrac, alpha=1;

  // indices identifying number and starting position of the different output // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const vector<Uint> net_outputs, net_indices, pol_start, adv_start;
  const Uint VsID = net_indices[0];

  // used in case of temporally correlated noise
  vector<Rvec> OrUhState = vector<Rvec>( nAgents, Rvec(nA, 0) );

  vector<float> outBuf;

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
  inline Advantage_t prepare_advantage(const Rvec& out,
      const Policy_t*const pol) const {
    return Advantage_t(adv_start, &aInfo, out, pol);
  }

  void prepareData()
  {
    // Rewards second moment is computed right before actual training begins
    // therefore we need to recompute (rescaled) Retrace values for all obss
    // seen before this point.
    Learner_offPolicy::prepareData();
    if(updatePrepared && nStep == 0) {
      #pragma omp parallel for schedule(dynamic)
      for(Uint i = 0; i < data->Set.size(); i++)
        for (Uint j=data->Set[i]->ndata(); j>0; j--)
          updateQret(data->Set[i], j, data->Set[i]->action_adv[j],
            data->Set[i]->state_vals[j], 1);
    }
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override
  {
    Sequence* const traj = data->Set[seq];
    const int ndata = traj->tuples.size()-1;
    if(thrID==1) profiler->stop_start("FWD");

    F[0]->prepare_seq(traj, thrID);
    for (int k=0; k<ndata; k++) F[0]->forward(traj, k, thrID);
    //if partial sequence then compute value of last state (!= R_end)
    if( traj->isTerminal(ndata) ) updateQret(traj, ndata, 0, 0, 0);
    else if( traj->isTruncated(ndata) ) {
      const Rvec nxt = F[0]->forward(traj, ndata, thrID);
      traj->setStateValue(ndata, nxt[VsID]);
      updateQret(traj, ndata, 0, nxt[VsID], 0);
    }

    if(thrID==1)  profiler->stop_start("POL");
    for(int k=ndata-1; k>=0; k--)
    {
      const Rvec out_cur = F[0]->get(traj, k, thrID);
      const Policy_t pol = prepare_policy(out_cur, traj->tuples[k]);
      #ifdef DKL_filter
        const bool isOff = traj->distFarPolicy(k, pol.sampKLdiv, CmaxRet-1);
      #else
        const bool isOff = traj->isFarPolicy(k, pol.sampImpWeight, CmaxRet);
      #endif
      // in case rho outside bounds, do not compute gradient
      Rvec G;
      #if RACER_SKIP == 1
        if(isOff) {
          G = offPolCorrUpdate(traj, k, out_cur, pol, thrID);
          continue;
        } else
      #endif
          G = compute(traj,k, out_cur, pol, thrID);
      //write gradient onto output layer:
      F[0]->backward(G, k, thrID);
    }

    if(thrID==1)  profiler->stop_start("BCK");
    F[0]->gradient(thrID);
  }

  void Train(const Uint seq, const Uint samp, const Uint thrID) const override
  {
    Sequence* const traj = data->Set[seq];
    assert(samp+1 < traj->tuples.size());

    if(thrID==1) profiler->stop_start("FWD");

    F[0]->prepare_one(traj, samp, thrID); // prepare thread workspace
    const Rvec out_cur = F[0]->forward(traj, samp, thrID); // network compute

    if( traj->isTerminal(samp+1) ) updateQret(traj, samp+1, 0, 0, 0);
    else if( traj->isTruncated(samp+1) ) {
      const Rvec nxt = F[0]->forward(traj, samp+1, thrID);
      traj->setStateValue(samp+1, nxt[VsID]);
      updateQret(traj, samp+1, 0, nxt[VsID], 0);
    }

    const Policy_t pol = prepare_policy(out_cur, traj->tuples[samp]);
    // check whether importance weight is in 1/Cmax < c < Cmax
    const bool isOff = traj->isFarPolicy(samp, pol.sampImpWeight, CmaxRet);

    if(thrID==1)  profiler->stop_start("CMP");
    Rvec grad;

    #if RACER_SKIP == 1
      if(isOff) grad = offPolCorrUpdate(traj, samp, out_cur, pol, thrID);
      else
    #endif
        grad = compute(traj, samp, out_cur, pol, thrID);

    if(thrID==1)  profiler->stop_start("BCK");
    F[0]->backward(grad, samp, thrID); // place gradient onto output layer
    F[0]->gradient(thrID);  // backprop
  }

  inline Rvec compute(Sequence*const traj, const Uint samp,
    const Rvec& outVec, const Policy_t& POL, const Uint thrID) const
  {
    const Advantage_t ADV = prepare_advantage(outVec, &POL);
    const Real A_cur = ADV.computeAdvantage(POL.sampAct), V_cur = outVec[VsID];
    // shift retrace-advantage with current V(s) estimate:
    const Real A_RET = traj->Q_RET[samp] +traj->state_vals[samp]-V_cur;
    const Real rho = POL.sampImpWeight, dkl = POL.sampKLdiv;
    const Real Ver = std::min((Real)1, rho) * (A_RET-A_cur);
    const Real Aer = std::min(CmaxRet, rho) * (A_RET-A_cur);
    const Rvec polG = policyGradient(traj->tuples[samp], POL,ADV,A_RET, thrID);
    const Rvec penalG  = POL.div_kl_grad(traj->tuples[samp]->mu, -1);
    const Rvec finalG  = weightSum2Grads(polG, penalG, beta);
    //prepare Q with off policy corrections for next step:
    const Real dAdv = updateQret(traj, samp, A_cur, V_cur, POL);

    Rvec gradient(F[0]->nOutputs(), 0);
    gradient[VsID] = beta*alpha * Ver;
    POL.finalize_grad(finalG, gradient);
    ADV.grad(POL.sampAct, beta*alpha * Aer, gradient);
    trainInfo->log(A_cur, A_RET-A_cur, polG, penalG, {beta, dAdv, rho}, thrID);
    traj->setMseDklImpw(samp, Ver*Ver, dkl, rho);
    return gradient;
  }

  inline Rvec offPolCorrUpdate(Sequence*const S, const Uint t,
    const Rvec output, const Policy_t& pol, const Uint thrID) const
  {
    const Advantage_t adv = prepare_advantage(output, &pol);
    const Real A_cur = adv.computeAdvantage(pol.sampAct);
    // shift retrace-advantage with current V(s) estimate:
    const Real A_RET = S->Q_RET[t] +S->state_vals[t] -output[VsID];
    const Real Ver = std::min((Real)1, pol.sampImpWeight) * (A_RET-A_cur);
    updateQret(S, t, A_cur, output[VsID], pol);
    S->setMseDklImpw(t, Ver*Ver, pol.sampKLdiv, pol.sampImpWeight);
    Rvec gradient(F[0]->nOutputs(), 0);
    const Rvec pg = pol.div_kl_grad(S->tuples[t]->mu, beta-1);
    pol.finalize_grad(pg, gradient);
    return gradient;
  }

  inline void updateQret(Sequence*const S, const Uint t) const {
    const Real rho = S->isLast(t) ? 0 : S->offPolicImpW[t];
    updateQret(S, t, S->action_adv[t], S->state_vals[t], rho);
  }
  inline void updateQretBack(Sequence*const S, const Uint t) const {
    if(t == 0) return;
    const Real W=S->isLast(t)? 0:S->offPolicImpW[t], R=data->scaledReward(S,t);
    const Real delta = R +gamma*S->state_vals[t] -S->state_vals[t-1];
    S->Q_RET[t-1] = delta + gamma*(W>1? 1:W)*(S->Q_RET[t] - S->action_adv[t]);
  }

  inline Real updateQret(Sequence*const S, const Uint t, const Real A,
    const Real V, const Policy_t& pol) const {
    // shift retrace advantage with update estimate for V(s_t)
    S->setRetrace(t, S->Q_RET[t] + S->state_vals[t] -V );
    S->setStateValue(t, V); S->setAdvantage(t, A);
    //prepare Qret_{t-1} with off policy corrections for future use
    return updateQret(S, t, A, V, pol.sampImpWeight);
  }

  inline Real updateQret(Sequence*const S, const Uint t, const Real A,
    const Real V, const Real rho) const {
    assert(rho >= 0);
    if(t == 0) return 0;
    const Real oldRet = S->Q_RET[t-1], W = rho>1 ? 1 : rho;
    const Real delta = data->scaledReward(S,t) +gamma*V - S->state_vals[t-1];
    S->setRetrace(t-1, delta + gamma*W*(S->Q_RET[t] - A) );
    return std::fabs(S->Q_RET[t-1] - oldRet);
  }

  inline Rvec policyGradient(const Tuple*const _t, const Policy_t& POL,
    const Advantage_t& ADV, const Real A_RET, const Uint thrID) const {
    const Real rho_cur = POL.sampImpWeight;
    #if defined(RACER_TABC)
      //compute quantities needed for trunc import sampl with bias correction
      const Action_t sample = POL.sample(&generators[thrID]);
      const Real polProbOnPolicy = POL.evalLogProbability(sample);
      const Real polProbBehavior = Policy_t::evalBehavior(sample, _t->mu);
      const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
      const Real A_pol = ADV.computeAdvantage(sample);
      const Real gain1 = A_RET*std::min((Real)1, rho_cur);
      const Real gain2 = A_pol*std::max((Real)0, 1-1/rho_pol);

      const Rvec gradAcer_1 = POL.policy_grad(POL.sampAct, gain1);
      const Rvec gradAcer_2 = POL.policy_grad(sample,      gain2);
      return sum2Grads(gradAcer_1, gradAcer_2);
    #else
    //min(CmaxRet,rho) stabilize if resampling is disabled (*will show
    // warning on screen*). No effect if functioning normally.
      return POL.policy_grad(POL.sampAct, A_RET*std::min(CmaxRet,rho_cur));
    #endif
  }

  //inline Rvec criticGrad(const Policy_t& POL, const Advantage_t& ADV,
  //  const Real A_RET, const Real A_critic) const {
  //  const Real anneal = iter()>epsAnneal ? 1 : Real(iter())/epsAnneal;
  //  const Real varCritic = ADV.advantageVariance();
  //  const Real iEpsA = std::pow(A_RET-A_critic,2)/(varCritic+2.2e-16);
  //  const Real eta = anneal * safeExp( -0.5*iEpsA);
  //  return POL.control_grad(&ADV, eta);
  //}

 public:
  RACER(Environment*const _env, Settings& _set, vector<Uint> net_outs,
    vector<Uint> pol_inds, vector<Uint> adv_inds) :
    Learner_offPolicy(_env, _set), tgtFrac(_set.penalTol),
    net_outputs(net_outs), net_indices(count_indices(net_outs)),
    pol_start(pol_inds), adv_start(adv_inds) {
    printf("RACER starts: v:%u pol:%s adv:%s\n", VsID,
    print(pol_start).c_str(), print(adv_start).c_str());
    trainInfo = new TrainData("racer", _set, 1, "| beta | dAdv | avgW ", 3);
  }
  ~RACER() { }

  void select(const Agent& agent) override
  {
    const int thrID= omp_get_thread_num();
    Sequence* const traj = data->inProgress[agent.ID];
    data->add_state(agent);

    if( agent.Status < TERM_COMM ) // not end of sequence
    {
      //Compute policy and value on most recent element of the sequence. If RNN
      // recurrent connection from last call from same agent will be reused
      Rvec output = F[0]->forward_agent(traj, agent, thrID);
      Policy_t pol = prepare_policy(output);
      const Advantage_t adv = prepare_advantage(output, &pol);
      Rvec mu = pol.getVector(); // vector-form current policy for storage

      // if explNoise is 0, we just act according to policy
      // since explNoise is initial value of diagonal std vectors
      // this should only be used for evaluating a learned policy
      Action_t act = pol.finalize(explNoise>0, &generators[thrID], mu);

      const Real advantage = adv.computeAdvantage(pol.sampAct);
      traj->action_adv.push_back(advantage);
      traj->state_vals.push_back(output[VsID]);
      agent.a->set(act);

      #ifdef dumpExtra
        traj->add_action(agent.a->vals, mu);
        Rvec param = adv.getParam();
        assert(param.size() == nL);
        mu.insert(mu.end(), param.begin(), param.end());
        agent.writeData(learn_rank, mu);
      #else
        data->add_action(agent, mu);
      #endif

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
        traj->state_vals.push_back(output[VsID]); // not a terminal state
      } else {
        traj->state_vals.push_back(0); //value of terminal state is 0
      }
      //whether seq is truncated or terminated, act adv is undefined:
      traj->action_adv.push_back(0);
      // compute initial Qret for whole trajectory:
      writeOnPolRetrace(traj, thrID);
      OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
      #ifdef dumpExtra
        agent.a->set(Rvec(nA,0));
        traj->add_action(agent.a->vals, Rvec(policyVecDim,0));
        agent.writeData(learn_rank, Rvec(policyVecDim+nL, 0));
        data->push_back(agent.ID);
      #else
        data->terminate_seq(agent);
      #endif
    }
  }

  void writeOnPolRetrace(Sequence*const seq, const int thrID) {
    assert(seq->tuples.size() == seq->action_adv.size());
    assert(seq->tuples.size() == seq->state_vals.size());
    assert(seq->Q_RET.size()  == 0);
    const Uint N = seq->tuples.size();
    //within Retrace, we use the state_vals vector to write the Q retrace values
    seq->Q_RET.resize(N, 0);
    for (Uint i=N-1; i>0; i--) //update all q_ret before terminal step
      updateQret(seq, i, seq->action_adv[i], seq->state_vals[i], 1);
  }

  void prepareGradient()
  {
    #ifdef RACER_BACKWARD
    if(updateComplete) {
      profiler->stop_start("QRET");
      #pragma omp parallel for schedule(dynamic)
      for(Uint i = 0; i < data->Set.size(); i++)
        for(int j = data->Set[i]->just_sampled-1; j > 0; j--)
          updateQretBack(data->Set[i], j);
    }
    #endif

    Learner_offPolicy::prepareGradientReFER();
  }
};

/*
  ## FORWARD RACER

      #if RACER_FORWARD>0 // prepare thread workspace
        F[0]->prepare(RACER_FORWARD+1, traj, samp, thrID);
      #else
      #endif
      #ifdef DKL_filter
        const Real KLdiv = pol.kl_divergence(S->tuples[t]->mu);
        const bool isOff = traj->distFarPolicy(t, KLdiv, 1+KLdiv, CmaxRet-1);
      #else

    #if RACER_FORWARD>0
      // do N steps of fwd net to obtain better estimate of Qret
      Uint N = std::min(traj->ndata()-samp, (Uint)RACER_FORWARD);
      for(Uint k = samp+1; k<=samp+N; k++)
      {
        if( traj->isTerminal(k) ) {
          assert(traj->action_adv[k] == 0 && traj->state_vals[k] == 0);
        } else if( traj->isTruncated(k) ) {
          assert(traj->action_adv[k] == 0);
          const Rvec nxt = F[0]->forward(traj, k, thrID);
          traj->setStateValue(k, nxt[VsID]);
        } else {
          const Rvec nxt = F[0]->forward(traj, k, thrID);
          const Policy_t polt = prepare_policy(nxt, traj->tuples[k]);
          const Advantage_t advt = prepare_advantage(nxt, &polt);
          //these are all race conditions:
          traj->setSquaredError(k, polt.kl_divergence(traj->tuples[k]->mu) );
          traj->setAdvantage(k, advt.computeAdvantage(polt.sampAct) );
          traj->setOffPolWeight(k, polt.sampImpWeight );
          traj->setStateValue(k, nxt[VsID] );
        }
      }
      for(Uint j = samp+N; j>samp; j--) updateQret(traj,j);
    #endif

  ## ADV DUMPING (bottom of writeOnPolRetrace)
  #if 0
    #pragma omp critical
    if(nStep>0) {
      // outbuf contains
      // - R[t] = sum_{t'=t}^{T-1} gamma^{t'-t} r_{t+1} (if seq is truncated
      //   instead of terminated, we must add V_T * gamma^(T-t) )
      // - Q^w(s_t,a_t) and Q^ret_t
      outBuf = vector<float>(4*(N-1), 0);
      for(Uint i=N-1; i>0; i--) {
        Real R = data->scaledReward(seq, i) +
          (seq->isTruncated(i)? gamma*seq->state_vals[i] : 0);
        for(Uint j = i; j>0; j--) { // add disc rew to R_t of all prev steps
          outBuf[4*(j-1) +0] += R; R *= gamma;
        }
        outBuf[4*(i-1) +1] = seq->action_adv[i-1];
        // we are actually storing A_RET in there:
        outBuf[4*(i-1) +2] = seq->Q_RET[i-1];
        outBuf[4*(i-1) +3] = seq->state_vals[i-1];
      }
      // revert scaling of rewards
      //for(Uint i=0; i<outBuf.size(); i--) outBuf[i] /= data->invstd_reward;
    }
  #endif
*/
