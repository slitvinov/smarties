/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

#define impSampVal
//#define dumpExtra
#ifndef RACER_SKIP
#define RACER_SKIP 1
#endif
//#define RACER_ONESTEPADV
#define RACER_BACKWARD
//#ifndef RACER_FORWARD
#define RACER_FORWARD 0
//#endif

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
  // learnR: net's learning rate
  // invC: inverse of c_max
  // alpha: weight of value-update relative to policy update. 1 means equal
  const Real tgtFrac_param, learnR, invC=1./CmaxRet, alpha=1;
  // indices identifying number and starting position of the different output // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const vector<Uint> net_outputs, net_indices, pol_start, adv_start;
  const Uint VsID = net_indices[0];

  // used in case of temporally correlated noise
  vector<Rvec> OrUhState = vector<Rvec>( nAgents, Rvec(nA, 0) );

  // used for debugging purposes to dump stats about gradient. will be removed
  FILE * wFile = fopen("grads_dist.raw", "ab");

  //tracks statistics about gradient, used for gradient clipping:
  StatsTracker* opcInfo;

  // initial value of relative weight of penalization to update gradients:
  Real beta = 0.2;

  MPI_Request nData_request = MPI_REQUEST_NULL;
  double ndata_reduce_result[2], ndata_partial_sum[2];

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
    Learner_offPolicy::prepareData();
    if(updatePrepared && nStep == 0) {
      if(!learn_rank) cout<<"Initial reward std "<<1/data->invstd_reward<<endl;
      #pragma omp parallel for schedule(dynamic)
      for(Uint i = 0; i < data->Set.size(); i++)
        for (Uint j=data->Set[i]->ndata(); j>0; j--)
          updateQret(data->Set[i], j, data->Set[i]->action_adv[j],
            data->Set[i]->state_vals[j], 1);
    }
  }

  void Train_BPTT(const Uint seq, const Uint thrID) const override
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
      const bool isOff = traj->isOffPolicy(k, pol.sampImpWeight, CmaxRet, invC);
      // in case rho outside bounds, do not compute gradient
      Rvec G;
      #if   RACER_SKIP == 1
        if(isOff) {
          offPolCorrUpdate(traj, k, out_cur, pol);
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

    #if RACER_FORWARD>0
      F[0]->prepare_opc(traj, samp, thrID); // prepare thread workspace
    #else
      F[0]->prepare_one(traj, samp, thrID); // prepare thread workspace
    #endif
    const Rvec out_cur = F[0]->forward(traj, samp, thrID); // network compute

    if( traj->isTerminal(samp+1) ) updateQret(traj, samp+1, 0, 0, 0);
    else if( traj->isTruncated(samp+1) ) {
      const Rvec nxt = F[0]->forward(traj, samp+1, thrID);
      traj->setStateValue(samp+1, nxt[VsID]);
      updateQret(traj, samp+1, 0, nxt[VsID], 0);
    }

    const Policy_t pol = prepare_policy(out_cur, traj->tuples[samp]);
    // check whether importance weight is in 1/Cmax < c < Cmax
    const bool isOff = traj->isOffPolicy(samp, pol.sampImpWeight, CmaxRet,invC);

    #if RACER_FORWARD>0
      // do N steps of fwd net to obtain better estimate of Qret
      Uint N = std::min(traj->ndata()-1-samp, (Uint)RACER_FORWARD);
      for(Uint k = samp+1; k<=samp+N; k++) { // && k<traj->ndata()
        const Rvec outt = F[0]->forward(traj, k, thrID);
        const Policy_t polt = prepare_policy(outt, traj->tuples[k]);
        const Advantage_t advt = prepare_advantage(outt, &polt);
        //these are all race conditions:
        traj->setSquaredError(k, polt.kl_divergence(traj->tuples[k]->mu) );
        traj->setAdvantage(k, advt.computeAdvantage(polt.sampAct) );
        traj->setOffPolWeight(k, polt.sampImpWeight );
        traj->setStateValue(k, outt[VsID] );
        //if (impW < 0.1) break;
      }
      for(Uint j = samp+N; j>samp; j--) updateQret(traj,j);
    #endif

    if(thrID==1)  profiler->stop_start("CMP");
    Rvec grad;

    #if   RACER_SKIP == 1
      if(isOff) { // only update stored offpol weight and qret and so on
        offPolCorrUpdate(traj, samp, out_cur, pol);
        // The correct behavior here is to resample. To avoid code hanging due
        // to bad choice of hyperparams there is this failsafe mechanism.
        // If this is triggered, warning will be printed to screen.
        if( beta > 0.05 ) return resample(thrID);
        else // if beta too small, grad \approx penalization gradient
          grad = offPolGrad(traj, samp, out_cur, pol, thrID);
      } else
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
    const Real rho_cur = POL.sampImpWeight;
    const Real Ver = beta*alpha*std::min((Real)1, rho_cur) * (A_RET-A_cur);
    //const Real Ver = beta*alpha * rho_cur * Q_dist;

    //const Real Aer = beta*alpha*Q_dist;
    const Real Aer = beta*alpha * rho_cur * (A_RET-A_cur);

    const Rvec polG = policyGradient(traj->tuples[samp], POL,ADV,A_RET, thrID);
    const Rvec penalG  = POL.div_kl_grad(traj->tuples[samp]->mu, -1);
    const Rvec finalG  = weightSum2Grads(polG, penalG, beta);

    #ifdef dumpExtra
      if(thrID == 1) {
        const float dist = POL.kl_divergence(traj->tuples[samp]->mu);
        float normT = 0, dot = 0, normG = 0, impW = POL.sampImpWeight;
        for(Uint i = 0; i < polG.size(); i++) {
          normG +=    polG[i] * polG[i];
          dot   +=    polG[i] *  penalG[i];
          normT +=  penalG[i] *  penalG[i];
        }
        float ret[]={dot/std::sqrt(normT), std::sqrt(normG), dist, impW};
        fwrite(ret, sizeof(float), 4, wFile);
      }
    #endif

    Rvec gradient(F[0]->nOutputs(), 0);
    gradient[VsID] = Ver;
    POL.finalize_grad(finalG, gradient);
    ADV.grad(POL.sampAct, Aer, gradient);

    Vstats[thrID].dumpStats(A_cur, A_RET-A_cur); //bookkeeping
    //prepare Q with off policy corrections for next step:
    const Real dAdv = updateQret(traj, samp, A_cur, V_cur, POL);
    Rvec sampleInfo {0, 0, 0, dAdv, POL.sampImpWeight};
    for(Uint i=0; i<polG.size(); i++) {
      sampleInfo[0] += std::fabs(polG[i]);
      sampleInfo[1] += std::fabs( penalG[i]);
      sampleInfo[2] += polG[i]*penalG[i];
    }
    opcInfo->track_vector(sampleInfo, thrID);
    return gradient;
  }

  inline Rvec offPolGrad(Sequence*const S, const Uint t,
    const Rvec output, const Policy_t& pol, const Uint thrID) const {
    // prepare penalization gradient:
    Rvec gradient(F[0]->nOutputs(), 0);
    const Rvec pg = pol.div_kl_grad(S->tuples[t]->mu, beta-1);
    pol.finalize_grad(pg, gradient);
    return gradient;
  }

  inline void offPolCorrUpdate(Sequence*const S, const Uint t,
    const Rvec output, const Policy_t& pol) const {
    const Advantage_t adv = prepare_advantage(output, &pol);
    updateQret(S, t, adv.computeAdvantage(pol.sampAct), output[VsID], pol);
  }

  inline void updateQret(Sequence*const S, const Uint t) const {
    updateQret(S, t, S->action_adv[t], S->state_vals[t], S->offPol_weight[t]);
  }

  inline Real updateQret(Sequence*const S, const Uint t, const Real A,
    const Real V, const Policy_t& pol) const {
    S->setSquaredError(t, pol.kl_divergence(S->tuples[t]->mu) );
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
      const Real gain1 = A_RET*std::min((Real) CmaxPol, rho_cur);
      const Real gain2 = A_pol*std::max((Real) 0, 1-CmaxPol/rho_pol);

      const Rvec gradAcer_1 = POL.policy_grad(POL.sampAct, gain1);
      const Rvec gradAcer_2 = POL.policy_grad(sample,      gain2);
      return sum2Grads(gradAcer_1, gradAcer_2);
    #else
      return POL.policy_grad(POL.sampAct, A_RET*rho_cur);
    #endif
  }

  inline Rvec criticGrad(const Policy_t& POL, const Advantage_t& ADV,
    const Real A_RET, const Real A_critic) const {
    const Real anneal = iter()>epsAnneal ? 1 : Real(iter())/epsAnneal;
    const Real varCritic = ADV.advantageVariance();
    const Real iEpsA = std::pow(A_RET-A_critic,2)/(varCritic+2.2e-16);
    const Real eta = anneal * safeExp( -0.5*iEpsA);
    return POL.control_grad(&ADV, eta);
  }

 public:
  RACER(Environment*const _env, Settings& _set, vector<Uint> net_outs,
    vector<Uint> pol_inds, vector<Uint> adv_inds) :
    Learner_offPolicy(_env, _set), tgtFrac_param(_set.klDivConstraint),
    learnR(_set.learnrate), net_outputs(net_outs),
    net_indices(count_indices(net_outs)),
    pol_start(pol_inds), adv_start(adv_inds) {
    printf("RACER starts: v:%u pol:%s adv:%s\n", VsID,
    print(pol_start).c_str(), print(adv_start).c_str());
    opcInfo = new StatsTracker(5, "racer", _set, 100);
    //test();

    // Uncomment this line to keep the sequences with the minimum average DKL
    // instead of discarding the oldest sequences.
    // You probably do not want to tho because the DKL and the offPol imp
    // weight tend to be correlated. This means that if low DKL sequences are
    // discarded, low offPoll W will be over-represented in the mem buffer.
    // In general this means that sequences with worse-than-expected outcomes
    // will be over represented in the mem buffer.
    // This is because samples with a larger \rho generally have a larger
    // pol grad magnitude. Therefore are more strongly pushed away from mu.
    //MEMBUF_FILTER_ALGO = MAXERROR;
    MEMBUF_FILTER_ALGO = FARPOLFRAC;

    if(_set.maxTotSeqNum < _set.batchSize)  die("maxTotSeqNum < batchSize")
  }
  ~RACER() {
    fclose(wFile);
  }

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

      // if greedyEps is 0, we just act according to policy
      // since greedyEps is initial value of diagonal std vectors
      // this should only be used for evaluating a learned policy
      Action_t act = pol.finalize(greedyEps>0, &generators[thrID], mu);

      #if 0 // add and update temporally correlated noise
        act = pol.updateOrUhState(OrUhState[agent.ID], mu, act, iter());
      #endif

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
      writeOnPolRetrace(traj);
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

  void writeOnPolRetrace(Sequence*const seq) const {
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
    const bool bWasPrepareReady = updateComplete;

    Learner::prepareGradient();

    if(not bWasPrepareReady) return;

    #ifdef RACER_BACKWARD
      profiler->stop_start("QRET");
      #pragma omp parallel for schedule(dynamic)
      for(Uint i = 0; i < data->Set.size(); i++)
        for(int j = data->Set[i]->just_sampled-1; j>0; j--)
          updateQret(data->Set[i], j, data->Set[i]->action_adv[j],
            data->Set[i]->state_vals[j], data->Set[i]->offPol_weight[j]);
    #endif

    profiler->stop_start("PRNE");

    advanceCounters();
    data->prune(CmaxRet, MEMBUF_FILTER_ALGO);
    Real fracOffPol = data->nOffPol / (Real) data->nTransitions;

    profiler->stop_start("SLP");


    if (learn_size > 1) {
      const bool firstUpdate = nData_request == MPI_REQUEST_NULL;
      if(not firstUpdate) MPI_Wait(&nData_request, MPI_STATUS_IGNORE);

      // prepare an allreduce with the current data:
      ndata_partial_sum[0] = data->nOffPol;
      ndata_partial_sum[1] = data->nTransitions;
      // use result from prev AllReduce to update rewards (before new reduce).
      // Assumption is that the number of off Pol trajectories does not change
      // much each step. Especially because here we update the off pol W only
      // if an observation is actually sampled. Therefore at most this fraction
      // is wrong by batchSize / nTransitions ( ~ 0 )
      // In exchange we skip an mpi implicit barrier point.
      fracOffPol = ndata_reduce_result[0] / ndata_reduce_result[1];

      MPI_Iallreduce(ndata_partial_sum, ndata_reduce_result, 2, MPI_DOUBLE,
                     MPI_SUM, mastersComm, &nData_request);
      // if no reduction done, partial sums are meaningless
      if(firstUpdate) return;
    }

    const Real tgtFrac = tgtFrac_param / CmaxPol; // / (1 + nStep * ANNEAL_RATE)
    //#ifdef ANNEAL_LEARNR
    //  const Real learnRate = learnR / (1 + nStep * ANNEAL_RATE);
    //#else
      const Real learnRate = learnR;
    //#endif

    //if( fracOffPol > tgtFrac * std::cbrt(nA) )
    if(fracOffPol>tgtFrac) beta = (1-learnRate)*beta; // iter converges to 0
    else beta = learnRate +(1-learnRate)*beta; //fixed point iter converge to 1

    if( beta < 0.05 )
    warn("beta too low. Decrease learnrate and/or increase klDivConstraint.");
  }

  void getMetrics(ostringstream& buff) const {
    opcInfo->reduce_approx();
    buff<<" "<<std::setw(6)<<std::setprecision(3)<<beta;
    buff<<" "<<std::setw(6)<<std::setprecision(1)<<opcInfo->instMean[0];
    buff<<" "<<std::setw(6)<<std::setprecision(1)<<opcInfo->instMean[1];
    buff<<" "<<std::setw(6)<<std::setprecision(1)<<opcInfo->instMean[2];
    buff<<" "<<std::setw(6)<<std::setprecision(2)<<opcInfo->instMean[3];
    buff<<" "<<std::setw(6)<<std::setprecision(2)<<opcInfo->instMean[4];
  }
  void getHeaders(ostringstream& buff) const {
    // beta: coefficient of update gradient to penalization gradient:
    //       g = g_loss * beta + (1-beta) * g_penal
    // polG, penG : average norm of these gradients
    // proj : average norm of projection of polG along penG
    //        it is usually negative because curr policy should be as far as
    //        possible from behav. pol. in the direction of update
    // dAdv : average magnitude of Qret update
    // avgW : average importance weight
    buff <<"| beta | polG | penG | proj | dAdv | avgW ";
  }
};
