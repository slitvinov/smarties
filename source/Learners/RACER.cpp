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
//#define RACER_BACKWARD
#ifndef RACER_FORWARD
#define RACER_FORWARD 0
#endif

template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_offPolicy
{
 protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Uint nL = Advantage_t::compute_nL(&aInfo);
  const Real DKL_param, invC=1./CmaxRet, alpha=1;
  const vector<Uint> net_outputs, net_indices, pol_start, adv_start;
  const Uint VsID = net_indices[0];
  StatsTracker* opcInfo;
  Real DKL_coef = 0.2;

  MPI_Request nData_request = MPI_REQUEST_NULL;
  double ndata_reduce_result[2], ndata_partial_sum[2];

  inline Policy_t prepare_policy(const vector<Real>& out,
    const Tuple*const t = nullptr) const {
    Policy_t pol(pol_start, &aInfo, out);
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }
  inline Advantage_t prepare_advantage(const vector<Real>& out,
      const Policy_t*const pol) const {
    return Advantage_t(adv_start, &aInfo, out, pol);
  }

  void Train_BPTT(const Uint seq, const Uint thrID) const override
  {
    Sequence* const traj = data->Set[seq];
    const int ndata = traj->tuples.size()-1;
    F[0]->prepare_seq(traj, thrID);

    if(thrID==1) profiler->stop_start("FWD");
    for (int k=0; k<ndata; k++) F[0]->forward(traj, k, thrID);

    //if partial sequence then compute value of last state (!= R_end)
    assert(traj->ended);
    traj->Q_RET[ndata] = data->standardized_reward(traj, ndata);

    if(thrID==1)  profiler->stop_start("POL");
    for(int k=ndata-1; k>=0; k--)
    {
      const vector<Real> out_cur = F[0]->get(traj, k, thrID);
      const Policy_t pol = prepare_policy(out_cur, traj->tuples[k]);
      const bool isOff = traj->isOffPolicy(k, pol.sampRhoWeight, CmaxRet, invC);
      // in case rho outside bounds, do not compute gradient
      vector<Real> G;
      #if   RACER_SKIP == 1
        if(isOff) {
          offPolCorrUpdate(traj, k, out_cur, pol);
          continue;
        } else
      #elif RACER_SKIP == 2
        if(isOff) {
          G = offPolGrad(traj, k, out_cur, pol, thrID);
        } else
      #endif
          G = compute(traj,k, out_cur, pol, thrID);
      //write gradient onto output layer:
      F[0]->backward(G, k, thrID);
    }

    if(thrID==1)  profiler->stop_start("BCK");
    F[0]->gradient(thrID);
    if(thrID==1)  profiler->stop_start("SLP");
  }

  void Train(const Uint seq, const Uint samp, const Uint thrID) const override
  {
    Sequence* const traj = data->Set[seq];
    assert(samp+1 < traj->tuples.size());
    if(samp+2 == traj->tuples.size()) // if sampled S_{T-1}, update qRet of s_T
      traj->Q_RET[samp+1] = data->standardized_reward(traj, samp+1);

    if(thrID==1) profiler->stop_start("FWD");

    #if RACER_FORWARD>0
      F[0]->prepare_opc(traj, samp, thrID);
    #else
      F[0]->prepare_one(traj, samp, thrID);
    #endif

    const vector<Real> out_cur = F[0]->forward(traj, samp, thrID);
    const Policy_t pol = prepare_policy(out_cur, traj->tuples[samp]);
    const bool isOff = traj->isOffPolicy(samp, pol.sampRhoWeight, CmaxRet,invC);

    #if RACER_FORWARD>0
      // do N steps of fwd net to obtain better estimate of Qret
      Uint N = std::min(traj->ndata()-1-samp, (Uint)RACER_FORWARD);
      for(Uint k = samp+1; k<=samp+N; k++) { // && k<traj->ndata()
        const vector<Real> outt = F[0]->forward(traj, k, thrID);
        const Policy_t polt = prepare_policy(outt, traj->tuples[k]);
        const Advantage_t advt = prepare_advantage(outt, &polt);
        //these are all race conditions:
        traj->SquaredError[k] = polt.kl_divergence_opp(traj->tuples[k]->mu);
        traj->action_adv[k] = advt.computeAdvantage(polt.sampAct);
        traj->offPol_weight[k] = polt.sampRhoWeight;
        traj->state_vals[k] = outt[VsID];
        //if (impW < 0.1) break;
      }
      for(Uint j = samp+N; j>samp; j--) updateQret(traj,j);
    #endif

    if(thrID==1)  profiler->stop_start("CMP");
    vector<Real> grad;

    #if   RACER_SKIP == 1
      if(isOff) {
        offPolCorrUpdate(traj, samp, out_cur, pol);
        return resample(thrID);
      } else
    #elif RACER_SKIP == 2
      if(isOff) {
        grad = offPolGrad(traj, samp, out_cur, pol, thrID);
      } else
    #endif
        grad = compute(traj, samp, out_cur, pol, thrID);

    if(thrID==1)  profiler->stop_start("BCK");
    F[0]->backward(grad, samp, thrID);
    F[0]->gradient(thrID);
    if(thrID==1)  profiler->stop_start("SLP");
  }

  inline vector<Real> compute(Sequence*const traj, const Uint samp,
    const vector<Real>& outVec, const Policy_t& pol_cur, const Uint thrID) const
  {
    vector<Real> gradient(F[0]->nOutputs(), 0);
    const Advantage_t adv_cur = prepare_advantage(outVec, &pol_cur);
    const Real A_cur = adv_cur.computeAdvantage(pol_cur.sampAct);
    const Real Q_RET = traj->Q_RET[samp+1], V_cur = outVec[VsID];
    const Real Q_dist = Q_RET -A_cur -V_cur, A_RET = Q_RET-V_cur;

    #ifdef impSampVal
      const Real fac = alpha*std::min((Real)1, pol_cur.sampRhoWeight);
    #else
      const Real fac = alpha;
    #endif
    const Real Ver = fac*Q_dist;

    #ifdef RACER_ONESTEPADV
      const Real rNext = data->standardized_reward(traj,samp+1);
      const Real vNext = traj->state_vals[samp+1];
      const Real Qer = alpha*(rNext + gamma*vNext -A_cur-V_cur);
    #else
      //const Real Qer = Ver;
      //const Real Qer = alpha*Q_dist;
      const Real Qer = alpha*pol_cur.sampRhoWeight * Q_dist;
    #endif

    const vector<Real> policyG = policyGradient(traj->tuples[samp], pol_cur,
      adv_cur, A_RET, thrID);
    const vector<Real> penal=pol_cur.div_kl_opp_grad(traj->tuples[samp]->mu,-1);
    const vector<Real> finalG = weightSum2Grads(policyG, penal, DKL_coef);

    #ifdef dumpExtra
      {
        Real normG=0, normT=numeric_limits<Real>::epsilon(), dot=0, normP=0;
        for(Uint i = 0; i < policyG.size(); i++) {
          normG += policyG[i] * policyG[i];
          dot   += policyG[i] *  penal[i];
          normT +=  penal[i] *  penal[i];
        }
        for(Uint i = 0; i < policyG.size(); i++)
          normP += std::pow(policyG[i] -dot*penal[i]/normT, 2);
        float R1 = sqrt(normG), R2 = sqrt(normT), R3 = dot/R2, R4 = sqrt(normP);
        vector<float> ret = {R1, R2, R3, R4};
        lock_guard<mutex> lock(buffer_mutex);
        FILE * wFile = fopen("grads_dist.raw", "ab");
        fwrite(ret.data(),sizeof(float),4,wFile); fflush(wFile); fclose(wFile);
      }
    #endif

    gradient[VsID] = Ver;
    pol_cur.finalize_grad(finalG, gradient);
    adv_cur.grad(pol_cur.sampAct, Qer, gradient);

    Vstats[thrID].dumpStats(A_cur, Q_dist); //bookkeeping
    //prepare Q with off policy corrections for next step:
    const Real dAdv = updateQret(traj, samp, A_cur, V_cur, pol_cur);
    vector<Real> sampleInfo {0, 0, 0, dAdv, pol_cur.sampImpWeight};
    for(Uint i=0; i<policyG.size(); i++) {
      sampleInfo[0] += std::fabs(policyG[i]);
      sampleInfo[1] += std::fabs(penal[i]);
      sampleInfo[2] += policyG[i]*penal[i];
    }
    opcInfo->track_vector(sampleInfo, thrID);
    return gradient;
  }

  inline vector<Real> offPolGrad(Sequence*const S, const Uint t,
    const vector<Real> output, const Policy_t& pol, const Uint thrID) const
  {
    const Advantage_t adv = prepare_advantage(output, &pol);
    updateQret(S, t, adv.computeAdvantage(pol.sampAct), output[VsID], pol);
    // prepare penalization gradient:
    vector<Real> gradient(F[0]->nOutputs(), 0);
    const vector<Real> pg = pol.div_kl_opp_grad(S->tuples[t]->mu, DKL_coef-1);
    pol.finalize_grad(pg, gradient);
    return gradient;
    //const Real r=data->standardized_reward(traj,t+1), v=traj->state_vals[t+1];
    //adv_cur.grad(act, r + gamma*v -A_cur-V_cur, gradient);
  }

  inline void offPolCorrUpdate(Sequence*const S, const Uint t,
    const vector<Real> output, const Policy_t& pol) const
  {
    const Advantage_t adv = prepare_advantage(output, &pol);
    updateQret(S, t, adv.computeAdvantage(pol.sampAct), output[VsID], pol);
  }

  inline void updateQret(Sequence*const S, const Uint t) const
  {
    const Real A = S->action_adv[t], V = S->state_vals[t];
    const Real R = data->standardized_reward(S, t);
    const Real W = std::min((Real)1, S->offPol_weight[t]);
    //prepare Qret with off policy corrections for next step:
    S->Q_RET[t] = R +gamma*(W*(S->Q_RET[t+1]-A-V) +V);
  }

  inline Real updateQret(Sequence*const S, const Uint t, const Real A,
    const Real V, const Policy_t& pol) const
  {
    const Real oldQret = S->Q_RET[t], C = S->Q_RET[t+1]-A-V;
    const Real W = std::min((Real)1, pol.sampRhoWeight);
    //prepare Qret with off policy corrections for next step:
    S->SquaredError[t] = pol.kl_divergence_opp(S->tuples[t]->mu);
    S->Q_RET[t] = data->standardized_reward(S,t) +gamma*(W*C +V);
    S->action_adv[t] = A; S->state_vals[t] = V;
    return std::fabs(S->Q_RET[t] - oldQret);
  }

  inline vector<Real> policyGradient(const Tuple*const _t, const Policy_t& POL,
    const Advantage_t& ADV, const Real A_RET, const Uint thrID) const
  {
    const Real rho_cur = POL.sampRhoWeight;
    #if defined(RACER_TABC)
      //compute quantities needed for trunc import sampl with bias correction
      const Action_t sample = POL.sample(&generators[thrID]);
      const Real polProbOnPolicy = POL.evalLogProbability(sample);
      const Real polProbBehavior = Policy_t::evalBehavior(sample, _t->mu);
      const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
      const Real A_pol = ADV.computeAdvantage(sample);
      const Real gain1 = A_RET*std::min((Real) CmaxPol, rho_cur);
      const Real gain2 = A_pol*std::max((Real) 0, 1-CmaxPol/rho_pol);

      const vector<Real> gradAcer_1 = POL.policy_grad(POL.sampAct, gain1);
      const vector<Real> gradAcer_2 = POL.policy_grad(sample,      gain2);
      return sum2Grads(gradAcer_1, gradAcer_2);
    #else
      return POL.policy_grad(POL.sampAct, A_RET*std::min(CmaxPol, rho_cur) );
    #endif
  }

  inline vector<Real> criticGrad(const Policy_t& POL, const Advantage_t& ADV,
    const Real A_RET, const Real A_critic) const
  {
    const Real anneal = iter()>epsAnneal ? 1 : Real(iter())/epsAnneal;
    const Real varCritic = ADV.advantageVariance();
    const Real iEpsA = std::pow(A_RET-A_critic,2)/(varCritic+2.2e-16);
    const Real eta = anneal * safeExp( -0.5*iEpsA);
    return POL.control_grad(&ADV, eta);
  }

 public:
  RACER(Environment*const _env, Settings& _set, vector<Uint> net_outs,
    vector<Uint> pol_inds, vector<Uint> adv_inds) :
    Learner_offPolicy(_env, _set), DKL_param(_set.klDivConstraint),
    net_outputs(net_outs), net_indices(count_indices(net_outs)),
    pol_start(pol_inds), adv_start(adv_inds)
  {
    printf("RACER starts: v:%u pol:%s adv:%s\n", VsID,
    print(pol_start).c_str(), print(adv_start).c_str());
    opcInfo = new StatsTracker(5, "racer", _set, 100);
    //test();
    ALGO = MAXERROR;
    if(_set.maxTotSeqNum < _set.batchSize)  die("maxTotSeqNum < batchSize")
  }
  ~RACER() { }

  void select(const Agent& agent) override
  {
    const int thrID= omp_get_thread_num();
    Sequence* const traj = data->inProgress[agent.ID];
    data->add_state(agent);

    if( agent.Status != 2 )
    {
      //Compute policy and value on most recent element of the sequence. If RNN
      // recurrent connection from last call from same agent will be reused
      vector<Real> output = F[0]->forward_agent(traj, agent, thrID);
      Policy_t pol = prepare_policy(output);
      const Advantage_t adv = prepare_advantage(output, &pol);
      vector<Real> beta = pol.getBeta();

      const Action_t act = pol.finalize(greedyEps>0, &generators[thrID], beta);
      const Real advantage = adv.computeAdvantage(pol.sampAct);
      traj->action_adv.push_back(advantage);
      traj->state_vals.push_back(output[VsID]);
      agent.a->set(act);

      #ifdef dumpExtra
        data->inProgress[agent.ID]->add_action(agent.a->vals, beta);
        vector<Real> param = adv.getParam();
        assert(param.size() == nL);
        beta.insert(beta.end(), param.begin(), param.end());
        agent.writeData(learn_rank, beta);
      #else
        data->add_action(agent, beta);
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
      writeOnPolRetrace(traj);

      #ifdef dumpExtra
        agent.a->set(vector<Real>(nA,0));
        data->inProgress[agent.ID]->add_action(agent.a->vals, vector<Real>(policyVecDim, 0));
        agent.writeData(learn_rank, vector<Real>(policyVecDim+nL, 0));
        data->push_back(agent.ID);
      #else
        data->terminate_seq(agent);
      #endif
    }
  }

  void writeOnPolRetrace(Sequence*const seq) const
  {
    assert(seq->tuples.size() == seq->action_adv.size()+1);
    assert(seq->tuples.size() == seq->state_vals.size()+1);
    assert(seq->Q_RET.size() == 0);
    const Uint N = seq->tuples.size();
    //within Retrace, we use the state_vals vetor to write the Q retrace values
    seq->Q_RET.resize(N, 0);
    seq->state_vals.push_back(0);
    //terminal Q_ret = term reward with state val = 0:
    seq->Q_RET[N-1] = data->standardized_reward(seq, N-1);

    for (Uint i=N-1; i>0; i--) { //update all q_ret before terminal step
      const Real R = data->standardized_reward(seq, i-1);
      // formula for Q_RET given rho = 1 :
      seq->Q_RET[i-1] = R +gamma*(seq->Q_RET[i] - seq->action_adv[i-1]);
    }
  }

  void prepareGradient()
  {
    const bool bWasPrepareReady = updateComplete;

    #ifdef RACER_BACKWARD
      if(updateComplete) {
        profiler->stop_start("QRET");
        #pragma omp parallel for schedule(dynamic)
        for(Uint i = 0; i < data->Set.size(); i++)
          for(int j = data->Set[i]->just_sampled; j>=0; j--) {
            const Real obsOpcW = data->Set[i]->offPol_weight[j];
            assert(obsOpcW >= 0);
            const Real R = data->standardized_reward(data->Set[i], j);
            const Real W = obsOpcW>1? 1 : obsOpcW;
            const Real A = data->Set[i]->action_adv[j];
            const Real V = data->Set[i]->state_vals[j];
            const Real Qret = data->Set[i]->Q_RET[j+1];
            data->Set[i]->Q_RET[j] = R +gamma*(W*(Qret -A-V)+V);
          }
        profiler->stop_start("SLP");
      }
    #endif

    Learner_offPolicy::prepareGradient();

    if(not bWasPrepareReady) return;

    // update sequences
    Real fracOffPol = data->nOffPol / data->nTransitions;
    profiler->stop_start("SLP");

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
    #ifdef RACER_ACERTRICK
    const Real tgtFrac = DKL_param/CmaxPol;
    #else
    const Real tgtFrac = DKL_param*std::sqrt(nA)/CmaxPol;
    #endif
    if(fracOffPol>tgtFrac) DKL_coef = .9999*DKL_coef;
    else DKL_coef = 1e-4 + .9999*DKL_coef;
  }

  void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const {
    //Learner_offPolicy::getMetrics(fileOut, screenOut);
    opcInfo->reduce_approx();
    screenOut<<" DKL:"<<DKL_coef<<" polStats:["<<print(opcInfo->instMean)
             <<"] clipRatio:"<<F[0]->gradStats->cutRatio;
    fileOut<<" "<<DKL_coef<<" "<<print(opcInfo->instMean)<<" "<<print(opcInfo->instStdv);
  }
};
