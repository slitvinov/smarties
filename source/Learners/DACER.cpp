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
#ifndef DACER_SKIP
#define DACER_SKIP 1
#endif
//#define DACER_ONESTEPADV
//#define DACER_BACKWARD
#ifndef DACER_FORWARD
#define DACER_FORWARD 0
#endif

template<typename Policy_t, typename Action_t>
class DACER : public Learner_offPolicy
{
 protected:
  // continuous actions: dimensionality of action vectors
  // discrete actions: number of options
  const Uint nA = Policy_t::compute_nA(&aInfo);

  // tgtFrac_param: target fraction of off-pol samples
  // learnR: net's learning rate
  // invC: inverse of c_max
  // alpha: weight of value-update relative to policy update. 1 means equal
  const Real learnR, tgtFrac_param, invC=1./CmaxRet, alpha=1;
  const Real retraceTrickPow = 1. / std::cbrt(nA);
  // indices identifying number and starting position of the different output // groups from the network, that are read by separate functions
  // such as state value, policy mean, policy std, adv approximator
  const vector<Uint> net_outputs, net_indices, pol_start;
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

  void prepareData()
  {
    Learner_offPolicy::prepareData();
    if(updatePrepared && nStep == 0) {
      cout << "Preparing initial sequences with computed reward 1/std " << data->invstd_reward << endl;
      #pragma omp parallel for schedule(dynamic)
      for(Uint i = 0; i < data->Set.size(); i++) {
        Sequence* const traj = data->Set[i];
        const int N = traj->ndata(); traj->setRetrace(N, 0);
        for(Uint j=N; j>0; j--) updateVret(traj, i-1, traj->state_vals[i-1], 1);
      }
    }
  }

  void Train_BPTT(const Uint seq, const Uint thrID) const override
  {
    die("");
  }

  void Train(const Uint seq, const Uint samp, const Uint thrID) const override
  {
    Sequence* const traj = data->Set[seq];
    assert(samp+1 < traj->tuples.size());

    if(thrID==1) profiler->stop_start("FWD");

    #if DACER_FORWARD>0
      F[0]->prepare_opc(traj, samp, thrID); // prepare thread workspace
    #else
      F[0]->prepare_one(traj, samp, thrID); // prepare thread workspace
    #endif

    const Rvec out_cur = F[0]->forward(traj, samp, thrID); // network compute
    const Policy_t pol = prepare_policy(out_cur, traj->tuples[samp]);
    // check whether importance weight is in 1/Cmax < c < Cmax
    const bool isOff = traj->isOffPolicy(samp, pol.sampRhoWeight, CmaxRet,invC);

    #if DACER_FORWARD>0
      // do N steps of fwd net to obtain better estimate of Qret
      Uint N = std::min(traj->ndata()-1-samp, (Uint)DACER_FORWARD);
      for(Uint k = samp+1; k<=samp+N; k++) { // && k<traj->ndata()
        const Rvec outt = F[0]->forward(traj, k, thrID);
        const Policy_t polt = prepare_policy(outt, traj->tuples[k]);
        //these are all race conditions:
        traj->setSquaredError(k, polt.kl_divergence_opp(traj->tuples[k]->mu) );
        traj->setOffPolWeight(k, polt.sampRhoWeight );
        traj->setStateValue(k, outt[VsID] );
        //if (impW < 0.1) break;
      }
      for(Uint j = samp+N; j>samp; j--) updateQret(traj,j);
    #endif

    if(thrID==1)  profiler->stop_start("CMP");
    Rvec grad;

    #if   DACER_SKIP == 1
      if(isOff) { // only update stored offpol weight and qret and so on
        updateVret(traj, samp, out_cur[VsID], pol);
        // correct behavior is to resample
        // to avoid bugs there is a failsafe mechanism
        // if that is triggered, warning will be printed to screen
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

  inline Rvec compute(Sequence*const S, const Uint t, const Rvec& outVec,
    const Policy_t& pol_cur, const Uint thrID) const
  {
    const Real rNext = data->standardized_reward(S, t+1), V_cur = outVec[VsID];
    const Real A_RET = rNext +gamma*(S->Q_RET[t+1]+S->state_vals[t+1]) - V_cur;
    const Real dAdv = updateVret(S, t, V_cur, pol_cur);
    const Real rho_cur = pol_cur.sampRhoWeight, Ver = S->Q_RET[t];

    const Rvec policyG = pol_cur.policy_grad(pol_cur.sampAct, A_RET*rho_cur);
    const Rvec penalG  = pol_cur.div_kl_opp_grad(S->tuples[t]->mu, -1);
    const Rvec finalG  = weightSum2Grads(policyG, penalG, beta);

    #ifdef dumpExtra
      if(thrID == 1) {
        const float dist = pol_cur.kl_divergence_opp(S->tuples[t]->mu);
        float normT = 0, dot = 0, normG = 0, impW = pol_cur.sampImpWeight;
        for(Uint i = 0; i < policyG.size(); i++) {
          normG += policyG[i] * policyG[i];
          dot   += policyG[i] *  penalG[i];
          normT +=  penalG[i] *  penalG[i];
        }
        float ret[]={dot/std::sqrt(normT), std::sqrt(normG), dist, impW};
        fwrite(ret, sizeof(float), 4, wFile);
      }
    #endif

    Rvec gradient(F[0]->nOutputs(), 0);
    gradient[VsID] = Ver;
    pol_cur.finalize_grad(finalG, gradient);

    Vstats[thrID].dumpStats(V_cur, Ver); //bookkeeping
    //prepare Q with off policy corrections for next step:

    Rvec sampleInfo {0, 0, 0, dAdv, pol_cur.sampImpWeight};
    for(Uint i=0; i<policyG.size(); i++) {
      sampleInfo[0] += std::fabs(policyG[i]);
      sampleInfo[1] += std::fabs( penalG[i]);
      sampleInfo[2] += policyG[i]*penalG[i];
    }
    opcInfo->track_vector(sampleInfo, thrID);
    return gradient;
  }

  inline Rvec offPolGrad(Sequence*const S, const Uint t, const Rvec output,
    const Policy_t& pol, const Uint thrID) const {
    // prepare penalization gradient:
    Rvec gradient(F[0]->nOutputs(), 0);
    const Rvec pg = pol.div_kl_opp_grad(S->tuples[t]->mu, beta-1);
    pol.finalize_grad(pg, gradient);
    return gradient;
  }

  inline Real updateVret(Sequence*const S, const Uint t, const Real V,
    const Policy_t& pol) const {
    S->setSquaredError(t, pol.kl_divergence_opp(S->tuples[t]->mu) );
    return updateVret(S, t, V, pol.sampRhoWeight);
  }

  inline Real updateVret(Sequence*const S, const Uint t, const Real V,
    const Real rho) const {
    const Real rNext = data->standardized_reward(S, t+1), oldVret = S->Q_RET[t];
    const Real vNext = S->state_vals[t+1], V_RET = S->Q_RET[t+1];
    const Real delta = rho * (rNext +gamma*vNext -V);
    const Real trace = gamma *.95 *std::pow(rho,retraceTrickPow) *(V_RET-vNext);
    //const Real trace = gamma *std::min((Real)1, rho) *(V_RET-vNext);
    S->setStateValue(t, V ); S->setRetrace(t, delta + trace);
    return std::fabs(S->Q_RET[t] - oldVret);
  }

 public:
  DACER(Environment*const _env, Settings& _set, vector<Uint> net_outs,
   vector<Uint> pol_inds): Learner_offPolicy(_env,_set), learnR(_set.learnrate),
   tgtFrac_param(_set.klDivConstraint), net_outputs(net_outs),
   net_indices(count_indices(net_outs)), pol_start(pol_inds)
  {
    printf("DACER starts: v:%u pol:%s\n", VsID, print(pol_start).c_str());
    opcInfo = new StatsTracker(5, "DACER", _set, 100);
    //test();

    // Uncomment this line to keep the sequences with the minimum average DKL.
    // You probably do not want to tho because we measured the correlation
    // coefficient between the DKL and the offPol imp weight to be 0.1 .
    // This is because samples with a larger imp. w generally have a larger
    // pol grad magnitude. Therefore are more strongly pushed away from mu.
    //FILTER_ALGO = MAXERROR;

    cout << CmaxPol << " " << CmaxRet << " " << invC << endl;
    if(_set.maxTotSeqNum < _set.batchSize)  die("maxTotSeqNum < batchSize")
  }
  ~DACER() {
    fclose(wFile);
  }

  void select(const Agent& agent) override
  {
    const int thrID= omp_get_thread_num();
    Sequence* const traj = data->inProgress[agent.ID];
    data->add_state(agent);

    if( agent.Status != 2 )
    {
      //Compute policy and value on most recent element of the sequence. If RNN
      // recurrent connection from last call from same agent will be reused
      Rvec output = F[0]->forward_agent(traj, agent, thrID);
      Policy_t pol = prepare_policy(output);
      Rvec mu = pol.getVector(); // vector-form current policy for storage

      // if greedyEps is 0, we just act according to policy
      // since greedyEps is initial value of diagonal std vectors
      // this should only be used for evaluating a learned policy
      Action_t act = pol.finalize(greedyEps>0, &generators[thrID], mu);

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
      writeOnPolRetrace(traj); // compute initial Qret for whole trajectory
      OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
      data->terminate_seq(agent);
    }
  }

  void writeOnPolRetrace(Sequence*const seq) const
  {
    assert(seq->tuples.size() == seq->state_vals.size()+1);
    assert(seq->Q_RET.size() == 0);
    const Uint N = seq->tuples.size();
    //within Retrace, we use the state_vals vector to write the Q retrace values
    seq->Q_RET.resize(N, 0);
    seq->state_vals.push_back(0); //value of terminal state is 0
    //TODO extend for non-terminal trajectories: one more v_state predict

    //terminal Q_ret = term reward with state val = 0:
    seq->Q_RET[N-1] = 0;

    //update all q_ret before terminal step
    for (Uint i=N-1; i>0; i--) updateVret(seq, i-1, seq->state_vals[i-1], 1);
  }

  void prepareGradient()
  {
    const bool bWasPrepareReady = updateComplete;

    #ifdef DACER_BACKWARD
      if(updateComplete) {
        profiler->stop_start("QRET");
        #pragma omp parallel for schedule(dynamic)
        for(Uint i = 0; i < data->Set.size(); i++)
          for(int j = data->Set[i]->just_sampled; j>=0; j--) {
            const Real obsOpcW = data->Set[i]->offPol_weight[j];
            assert(obsOpcW >= 0);
            updateVret(data->Set[i], j, data->Set[i]->state_vals[j], obsOpcW);
          }
        profiler->stop_start("SLP");
      }
    #endif

    Learner_offPolicy::prepareGradient();

    if(not bWasPrepareReady) return;

    // update sequences
    Real fracOffPol = data->nOffPol / (Real) data->nTransitions;

    if (learn_size > 1) {
      const bool firstUpdate = nData_request == MPI_REQUEST_NULL;
      if(not firstUpdate) MPI_Wait(&nData_request, MPI_STATUS_IGNORE);

      // prepare an allreduce with the current data:
      ndata_partial_sum[0] = data->nOffPol;
      ndata_partial_sum[1] = data->nTransitions;
      // use result from prev reduce to update rewards (before new reduce)
      fracOffPol = ndata_reduce_result[0] / ndata_reduce_result[1];

      MPI_Iallreduce(ndata_partial_sum, ndata_reduce_result, 2, MPI_DOUBLE, MPI_SUM, mastersComm, &nData_request);
      // if no reduction done, partial sums are meaningless
      if(firstUpdate) return;
    }

    const Real tgtFrac = tgtFrac_param / CmaxPol / (1 + nStep * ANNEAL_RATE);
    #ifdef ANNEAL_LEARNR
      const Real learnRate = learnR / (1 + nStep * ANNEAL_RATE);
    #else
      const Real learnRate = learnR;
    #endif

    if( fracOffPol > tgtFrac * std::cbrt(nA) )
      beta = (1-learnRate)*beta; // fixed point iter converges to 0
    else
      beta = learnRate +(1-learnRate)*beta; // fixed point iter converges to 1

    if( beta < 0.05 )
    warn("beta too low. Decrease learnrate and/or increase klDivConstraint.");
  }

  void getMetrics(ostringstream& buff) const
  {
    opcInfo->reduce_approx();
    buff<<" "<<std::setw(6)<<std::setprecision(3)<<beta;
    buff<<" "<<std::setw(6)<<std::setprecision(1)<<opcInfo->instMean[0];
    buff<<" "<<std::setw(6)<<std::setprecision(1)<<opcInfo->instMean[1];
    buff<<" "<<std::setw(6)<<std::setprecision(1)<<opcInfo->instMean[2];
    buff<<" "<<std::setw(6)<<std::setprecision(2)<<opcInfo->instMean[3];
    buff<<" "<<std::setw(6)<<std::setprecision(2)<<opcInfo->instMean[4];
  }
  void getHeaders(ostringstream& buff) const
  {
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
