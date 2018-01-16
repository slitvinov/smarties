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
//#define UNBR
//#define REALBND (Real)1
#define REALBND CmaxPol
//#define ACER_TRUSTREGION

template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_offPolicy
{
 protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Uint nL = Advantage_t::compute_nL(&aInfo);
  const Real DKL_param;
  Real DKL_coef = 0.2;
  const vector<Uint> net_outputs, net_indices;
  const vector<Uint> pol_start, adv_start;
  const Uint VsID = net_indices[0];
  StatsTracker* opcInfo;

  MPI_Request nData_request = MPI_REQUEST_NULL;
  double ndata_reduce_result[2], ndata_partial_sum[2];

  inline Policy_t prepare_policy(const vector<Real>& out) const
  {
    return Policy_t(pol_start, &aInfo, out);
  }
  inline Advantage_t prepare_advantage(const vector<Real>& out,
      const Policy_t*const pol) const
  {
    return Advantage_t(adv_start, &aInfo, out, pol);
  }
  inline Policy_t* new_policy(const vector<Real>& out) const
  {
    return new Policy_t(pol_start, &aInfo, out);
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
    traj->state_vals[ndata] = data->standardized_reward(traj, ndata);

    if(thrID==1)  profiler->stop_start("POL");
    for(int k=ndata-1; k>=0; k--)
    {
      const vector<Real> out_cur = F[0]->get(traj, k, thrID);
      Policy_t pol = prepare_policy(out_cur);

      Tuple * const _t = traj->tuples[k];
      pol.prepare(_t->a, _t->mu);
      traj->offPol_weight[k] = pol.sampImpWeight;

      #if 1 // in case rho outside bounds, do not compute gradient
      if(pol.sampImpWeight < std::min((Real)0.5, 1/CmaxPol) ||
         pol.sampImpWeight > std::max((Real)2.0,   CmaxPol) )  {
        offPolCorrUpdate(traj, k, out_cur, pol);
        continue;
      }
      #endif

      vector<Real> G = compute(traj,k, out_cur, pol, thrID);
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
      traj->state_vals[samp+1] = data->standardized_reward(traj, samp+1);

    if(thrID==1) profiler->stop_start("FWD");
    F[0]->prepare_one(traj, samp, thrID);
    const vector<Real> out_cur = F[0]->forward(traj, samp, thrID);
    Policy_t pol = prepare_policy(out_cur);

    pol.prepare(traj->tuples[samp]->a, traj->tuples[samp]->mu);
    traj->offPol_weight[samp] = pol.sampImpWeight;

   if(pol.sampImpWeight < std::min((Real)0.5, 1/CmaxPol) ||
      pol.sampImpWeight > std::max((Real)2.0,   CmaxPol) )
   {
     offPolCorrUpdate(traj, samp, out_cur, pol);
     if(thrID==1)  profiler->stop_start("SLP");
     return resample(thrID);
   }

    if(thrID==1)  profiler->stop_start("CMP");
    vector<Real> grad = compute(traj, samp, out_cur, pol, thrID);

    if(thrID==1)  profiler->stop_start("BCK");
    F[0]->backward(grad, samp, thrID);
    F[0]->gradient(thrID);
    if(thrID==1)  profiler->stop_start("SLP");
  }

  inline vector<Real> compute(Sequence*const traj, const Uint samp,
    const vector<Real>& outVec, const Policy_t& pol_cur, const Uint thrID) const
  {
    Real meanPena=0, meanGrad=0, meanProj=0, meanDist=0;
    vector<Real> gradient(F[0]->nOutputs(), 0);
    const Advantage_t adv_cur = prepare_advantage(outVec, &pol_cur);
    const Action_t& act = pol_cur.sampAct; //unbounded action space
    const Real A_cur = adv_cur.computeAdvantage(act);

    const Real Q_RET = traj->state_vals[samp+1], V_cur = outVec[VsID];
    const Real Q_dist = Q_RET -A_cur-V_cur, A_RET = Q_RET-V_cur;

    #ifdef impSampVal
      #ifdef UNBR
        const Real Ver = .1*         pol_cur.sampImpWeight          *Q_dist;
      #else
        const Real Ver = .1*std::min(pol_cur.sampImpWeight, REALBND)*Q_dist;
      #endif
    #else
      const Real Ver = 0.1*Q_dist;
    #endif

    const vector<Real> policyG = policyGradient(traj->tuples[samp], pol_cur,
      adv_cur, A_RET, thrID);

    const vector<Real> penal=pol_cur.div_kl_opp_grad(traj->tuples[samp]->mu,-1);
    const vector<Real> finalG = weightSum2Grads(policyG, penal, DKL_coef);

    for(Uint i=0; i<policyG.size(); i++) {
      meanGrad += std::fabs(policyG[i]);
      meanPena += std::fabs(penal[i]);
      meanProj += policyG[i]*penal[i];
      meanDist += std::fabs(policyG[i]-finalG[i]);
    }
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
    adv_cur.grad(act, Ver, gradient);
    pol_cur.finalize_grad(finalG, gradient);

    //prepare Q with off policy corrections for next step:
    const Real R = data->standardized_reward(traj, samp);
    const Real W = std::min((Real)1, pol_cur.sampImpWeight);
    traj->state_vals[samp] = R +gamma*(W*(Q_RET-A_cur-V_cur) +V_cur);

    //bookkeeping:
    Vstats[thrID].dumpStats(A_cur+V_cur, Q_dist);
    opcInfo->track_vector({meanGrad, meanPena, meanProj, meanDist}, thrID);
    traj->SquaredError[samp]=min(1/pol_cur.sampImpWeight,pol_cur.sampImpWeight);
    return gradient;
  }

  inline void offPolCorrUpdate(Sequence*const traj, const Uint samp,
    const vector<Real> output, const Policy_t& pol) const
  {
    const Real Q_RET = traj->state_vals[samp+1];
    const Advantage_t adv_cur = prepare_advantage(output, &pol);
    const Action_t& act = pol.sampAct; //off policy stored action
    const Real V_cur = output[VsID], A_cur = adv_cur.computeAdvantage(act);
    //prepare rolled Q with off policy corrections for next step:
    const Real R = data->standardized_reward(traj, samp);
    const Real W = std::min((Real)1, pol.sampImpWeight);
    traj->state_vals[samp] = R +gamma*(W*(Q_RET-A_cur-V_cur) +V_cur);
    traj->SquaredError[samp]=min(1/pol.sampImpWeight, pol.sampImpWeight); //*std::fabs(Q_RET-A_hat-V_hat);
  }

  inline vector<Real> policyGradient(const Tuple*const _t, const Policy_t& POL, const Advantage_t& ADV, const Real A_RET, const Uint thrID) const
  {
    const Real rho_cur = POL.sampImpWeight;
    #if   defined(ACER_TABC)
      //compute quantities needed for trunc import sampl with bias correction
      const Action_t sample = POL.sample(&generators[thrID]);
      const Real polProbOnPolicy = POL.evalLogProbability(sample);
      const Real polProbBehavior = Policy_t::evalBehavior(sample, _t->mu);
      const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
      const Real A_pol = ADV.computeAdvantage(sample);
      const Real gain1 = A_RET*std::min((Real) 5, rho_cur);
      const Real gain2 = A_pol*std::max((Real) 0, 1-5/rho_pol);

      const vector<Real> gradAcer_1 = POL.policy_grad(POL.sampAct, gain1);
      const vector<Real> gradAcer_2 = POL.policy_grad(sample,      gain2);
      return sum2Grads(gradAcer_1, gradAcer_2);
    #else
      #ifdef UNBR
        return POL.policy_grad(POL.sampAct, A_RET * rho_cur);
      #else
        return POL.policy_grad(POL.sampAct, A_RET * std::min(REALBND,rho_cur));
      #endif
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
    opcInfo = new StatsTracker(4, "racer", _set);
    //test();
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
      traj->action_adv.push_back(adv.computeAdvantage(pol.sampAct));
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
        Policy_t dbg = prepare_policy(output);
        dbg.prepare(traj->tuples.back()->a, traj->tuples.back()->mu);
        const double err = fabs(dbg.sampImpWeight-1);
        if(err>1e-10) _die("Imp W err %20.20e", err);
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
    assert(seq->state_vals.size() == 0);
    const Uint N = seq->tuples.size();
    //within Retrace, we use the state_vals vetor to write the Q retrace values
    seq->state_vals.resize(N, 0);
    //terminal Q_ret = term reward with state val = 0:
    seq->state_vals[N-1] = data->standardized_reward(seq, N-1);

    for (Uint i=N-1; i>0; i--) { //update all q_ret before terminal step
      const Real R = data->standardized_reward(seq, i-1);
      seq->state_vals[i-1] = R+gamma*(seq->state_vals[i]-seq->action_adv[i-1]);
    }
  }

  void prepareGradient()
  {
    const bool bWasPrepareReady = updateComplete;

    Learner_offPolicy::prepareGradient();

    if(not bWasPrepareReady) return;

    // update sequences
    Real fracOffPol = data->nOffPol / data->nTransitions;
    profiler->stop_start("SLP");

    if (learn_size > 1)
    {
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
    const Real tgtFrac = DKL_param/CmaxPol;
    if(fracOffPol>tgtFrac) DKL_coef = .9999*DKL_coef;
    else DKL_coef = 1e-4 + .9999*DKL_coef;
  }

  void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
  {
    //Learner_offPolicy::getMetrics(fileOut, screenOut);
    opcInfo->reduce_approx();
    screenOut<<" DKL:"<<DKL_coef<<" polStats:["<<print(opcInfo->avgVec[0])<<"]";
    fileOut<<" "<<DKL_coef<<" "<<print(opcInfo->avgVec[0])<<" "<<print(opcInfo->stdVec[0]);
  }
};
