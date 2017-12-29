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
  const Real CmaxPol, DKL_target_orig;
  Real DKL_target = DKL_target_orig;
  const vector<Uint> net_outputs, net_indices;
  const vector<Uint> pol_start, adv_start;
  const Uint VsID = net_indices[0];
  const Uint PenalID = net_indices.back(), QPrecID = net_indices.back()+1;
  StatsTracker* opcInfo;

  MPI_Request nData_request = MPI_REQUEST_NULL;
  double ndata_reduce_result, ndata_partial_sum;

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
    for (int k=0; k<ndata; k++) {
      F[0]->forward<CUR>(traj, k, thrID);
      #ifdef ACER_TARGETNET
        F[0]->forward<TGT>(traj, k, thrID);
      #endif
    }

    //if partial sequence then compute value of last state (!= R_end)
    assert(traj->ended);
    traj->state_vals[ndata] = data->standardized_reward(traj, ndata);

    if(thrID==1)  profiler->stop_start("POL");
    for(int k=ndata-1; k>=0; k--)
    {
      const vector<Real> out_cur = F[0]->get<CUR>(traj, k, thrID);
      Policy_t pol = prepare_policy(out_cur);
      #ifdef ACER_TARGETNET //predict samp with tgt w
        const vector<Real> out_hat = F[0]->get<TGT>(traj, k, thrID);
        const Policy_t pol_target = prepare_policy(out_hat);
        const Policy_t* const polTgt = &pol_target;
      #else
        const Policy_t* const polTgt = nullptr;
      #endif

      Tuple * const _t = traj->tuples[k];
      pol.prepare(_t->a, _t->mu);
      traj->offPol_weight[k] = pol.sampImpWeight;

      #if 1 // in case rho outside bounds, do not compute gradient
      if(pol.sampImpWeight < std::min((Real)0.5, 1/CmaxPol) ||
         pol.sampImpWeight > std::max((Real)2.0,   CmaxPol) )  continue;
      #endif

      vector<Real> G = compute(traj,k, out_cur, pol,polTgt, thrID);
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

    #if 1 // in case rho outside bounds, resample:
     if(pol.sampImpWeight < std::min((Real)0.5, 1/CmaxPol) ||
        pol.sampImpWeight > std::max((Real)2.0,   CmaxPol) )
     {
       offPolCorrUpdate(traj, samp, out_cur, pol);
       if(thrID==1)  profiler->stop_start("SLP");
       return resample(thrID);
     }
    #endif

    #ifdef ACER_TARGETNET //predict samp with tgt w
      const vector<Real> out_hat = F[0]->forward<TGT>(traj, samp, thrID);
      const Policy_t pol_target = prepare_policy(out_hat);
      const Policy_t* const polTgt = &pol_target;
    #else
      const Policy_t* const polTgt = nullptr;
    #endif

    if(thrID==1)  profiler->stop_start("CMP");
    vector<Real> grad = compute(traj, samp, out_cur, pol, polTgt, thrID);

    if(thrID==1)  profiler->stop_start("BCK");
    F[0]->backward(grad, samp, thrID);
    F[0]->gradient(thrID);
    if(thrID==1)  profiler->stop_start("SLP");
  }

  inline vector<Real> compute(Sequence*const traj, const Uint samp,
    const vector<Real>& outVec, const Policy_t& pol_cur,
    const Policy_t* const pol_hat, const Uint thrID) const
  {
    Real meanPena=0, meanGrad=0, meanProj=0, meanDist=0;
    vector<Real> gradient(F[0]->nOutputs(), 0);
    const Advantage_t adv_cur = prepare_advantage(outVec, &pol_cur);
    const Action_t& act = pol_cur.sampAct; //unbounded action space
    const Real A_cur = adv_cur.computeAdvantage(act);

    const Real Q_RET = traj->state_vals[samp+1];
    const Real V_cur = outVec[VsID];
    const Real Qprec = outVec[QPrecID];
    const Real penalDKL = outVec[PenalID];
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

    #if   defined(ACER_TARGETNET)
      const Real DivKL = pol_cur.kl_divergence_opp(pol_hat);
      const vector<Real> penalG = penalTarget(pol_cur, pol_hat, penalDKL, DivKL, gradient);
      const vector<Real> finalG = sum2Grads(policyG, penalG);
    #elif defined(ACER_TRUSTREGION)
      const Real DivKL = pol_cur.kl_divergence_opp(traj->tuples[samp]->mu);
      const vector<Real> penalG = pol_cur.div_kl_opp_grad(traj->tuples[samp]->mu, 1);
      const vector<Real> finalG =circle_region(policyG, penalG, nA, DKL_target);
    #else
      const vector<Real> penalG = penalSample(traj->tuples[samp], pol_cur, A_RET, penalDKL, gradient);
      //const vector<Real> finalG = sum2Grads(penalG, policyG);
      const vector<Real> finalG = weightSum2Grads(policyG, penalG, DKL_target);
    #endif

    for(Uint i=0; i<policyG.size(); i++) {
      meanGrad += std::fabs(policyG[i]);
      meanPena += std::fabs(penalG[i]);
      meanProj += policyG[i]*penalG[i];
      meanDist += std::fabs(policyG[i]-finalG[i]);
    }
    #ifdef dumpExtra
      {
        Real normG=0, normT=numeric_limits<Real>::epsilon(), dot=0, normP=0;
        for(Uint i = 0; i < policyG.size(); i++) {
          normG += policyG[i] * policyG[i];
          dot   += policyG[i] *  penalG[i];
          normT +=  penalG[i] *  penalG[i];
        }
        for(Uint i = 0; i < policyG.size(); i++)
          normP += std::pow(policyG[i] -dot*penalG[i]/normT, 2);
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
    //decrease prec if err is large, from d Dkl(Q^RET || Q_th)/dQprec
    gradient[QPrecID] = -.5*(Q_dist*Q_dist - 1/Qprec);

    //prepare Q with off policy corrections for next step:
    const Real R = data->standardized_reward(traj, samp);
    //const Real W = std::min((Real)1, pol_cur.sampImpWeight);
    const Real W = std::min((Real)1, pol_cur.sampRhoWeight);
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
    //const Real W = std::min((Real)1, pol.sampImpWeight);
    const Real W = std::min((Real)1, pol.sampRhoWeight);
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

  inline vector<Real> penalTarget(const Policy_t&POL, const Policy_t*const TGT,
    const Real penalDKL, const Real DivKL, vector<Real>&grad) const
  {
    //increase if DivKL is greater than Target
    //computed as \nabla_{penalDKL} (DivKL - DKL_target)^4
    //with rough approximation that d DivKL/ d penalDKL \propto penalDKL
    //(distance increases if penalty term increases, similar to PPO )
    grad[PenalID] = std::pow((DivKL-DKL_target)/DKL_target,3)*penalDKL;
    //gradient[PenalID] = (DivKL - DKL_target)*penalDKL/DKL_target;
    return POL.div_kl_opp_grad(TGT, -penalDKL);
  }

  inline vector<Real> penalSample(const Tuple*const _t, const Policy_t& POL,
  const Real A_RET, const Real penalDKL, vector<Real>& grad) const
  {
    const Real DKLmul = -1;
    //const Real DKLmul = -10*annealingFactor();
    #ifdef UNBR
    //const Real DKLmul= -max((Real)0, A_RET) *     POL.sampInvWeight;
    #else
    //const Real DKLmul= -max((Real)0, A_RET) * min(POL.sampInvWeight, REALBND);
    #endif
    //grad[PenalID] = (DivKL - 0.1)*penalDKL/0.1;
    //grad[PenalID] = std::pow((DivKL - 0.1)/0.1, 3)*penalDKL;
    return POL.div_kl_opp_grad(_t->mu, DKLmul);
  }

 public:
  RACER(Environment*const _env, Settings& sett, vector<Uint> net_outs,
    vector<Uint> pol_inds, vector<Uint> adv_inds) :
    Learner_offPolicy(_env, sett), CmaxPol(sett.impWeight),
    DKL_target_orig(sett.klDivConstraint), net_outputs(net_outs),
    net_indices(count_indices(net_outs)), pol_start(pol_inds), adv_start(adv_inds)
  {
    printf("RACER starts: v:%u pol:%s adv:%s penal:%u prec:%u\n", VsID,
    print(pol_start).c_str(), print(adv_start).c_str(), PenalID, QPrecID);
    opcInfo = new StatsTracker(4, "racer", sett);
    //test();
    if(sett.maxTotSeqNum < sett.batchSize)  die("maxTotSeqNum < batchSize")
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
    // this does not count ones added between updates by exploration
    //const Real lastSeq = nStoredSeqs_last;
    //const Real currPre = data->nSequences; //pre pruning
    //assert(nStoredSeqs_last <= data->nSequences); //before pruining
    profiler->stop_start("PRNE");
    const Real fracOffPol = data->prune2(CmaxPol, 256*batchSize);
    profiler->stop_start("SLP");

    if(fracOffPol>.01*std::sqrt(nA)) DKL_target = 0.9999*DKL_target;
    else DKL_target = 1e-4 + 0.9999*DKL_target;

    //const Real nPruned = currPre - currSeqs;
    //assuming that pruning has not removed any of the fresh samples
    //compute how many samples we would have in a purely sequential code:
    //Real samples_sequential = lastSeq - nPruned;
    nStoredSeqs_last = data->nSequences; //after pruning

    /*
    if (learn_size > 1)
    {
      const bool firstUpdate = nData_request == MPI_REQUEST_NULL;
      if(not firstUpdate) MPI_Wait(&nData_request, MPI_STATUS_IGNORE);

      // prepare an allreduce with the current data:
      ndata_partial_sum = samples_sequential;
      // use result from prev reduce to update rewards (before new reduce)
      samples_sequential = ndata_reduce_result;

      MPI_Iallreduce(&ndata_partial_sum, &ndata_reduce_result, 1, MPI_DOUBLE, MPI_SUM, mastersComm, &nData_request);
      // if no reduction done, partial sums are meaningless
      if(firstUpdate) return;
    }
    */
    //if(nPruned>0.5) DKL_target *= std::pow(0.999, nPruned);
    //else            DKL_target += 1e-3*(1-DKL_target); //0.001 + 0.999*DKL_target;
    //if(samples_sequential < nSequences4Train()*learn_size)
    //  DKL_target *= 0.99;
    //else
    //  DKL_target += 1e-5*(1-DKL_target);
  }

  void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
  {
    //Learner_offPolicy::getMetrics(fileOut, screenOut);
    opcInfo->reduce_approx();
    const Parameters*const W = F[0]->net->weights;
    const nnReal*const parameters = W->B(W->nLayers - 1);
    const Real Qprec=std::exp(parameters[1]), penalDKL=std::exp(parameters[0]);

    screenOut<<" DKL:["<<DKL_target<<" "<<penalDKL<<"] prec:"<<Qprec
        <<" polStats:["<<print(opcInfo->avgVec[0])<<"]";
    fileOut<<" "<<print(opcInfo->avgVec[0])<<" "<<print(opcInfo->stdVec[0]);
  }
};

#if 0
void Train(const Uint seq, const Uint samp, const Uint thrID) const override
{
  Sequence* const traj = data->Set[seq];
  //last state for which we compute a policy: T-1
  Uint lastTPolicy = traj->tuples.size() -2;
  bool truncated = not traj->ended;
  if(thrID==1) profiler->stop_start("FWD");

  vector<Policy_t> policies;
  policies.reserve(traj->tuples.size() -samp-1);

  F[0]->prepare_opc(traj, samp, thrID);
  const vector<Real> out_cur = F[0]->forward(traj, samp, thrID);
  policies.push_back(prepare_policy(out_cur));

  #ifdef ACER_TARGETNET //predict samp with tgt w
    const vector<Real> out_hat = F[0]->forward<TGT>(traj, samp, thrID);
    const Policy_t pol_target = prepare_policy(out_hat);
    const Policy_t* const polTgt = &pol_target;
  #else
    const Policy_t* const polTgt = nullptr;
  #endif

  policies[0].prepare(traj->tuples[samp]->a, traj->tuples[samp]->mu);
  traj->offPol_weight[samp] = policies[0].sampImpWeight;

  #if 1 // in case rho outside bounds, resample:
   if(policies[0].sampImpWeight < std::min((Real)0.5, 1/CmaxPol) ||
      policies[0].sampImpWeight > std::max((Real)2.0,   CmaxPol) ) {
     if(thrID==1)  profiler->stop_start("SLP");
     return resample(thrID);
   }
  #endif

  // initialize importance sample
  #ifdef UNBR
    Real impW = policies[0].sampImpWeight;
  #else
    Real impW = std::min(REALBND, policies[0].sampImpWeight);
  #endif

  //Compute off-pol corrections. Skip last state of seq: we need all V(snext)
  for(Uint k=samp+1; k<=lastTPolicy; k++)  {
    const vector<Real> out_tmp = F[0]->forward(traj, k, thrID);
    policies.push_back(prepare_policy(out_tmp));

    assert(policies.size() == k+1-samp);
    policies[k-samp].prepare(traj->tuples[k]->a, traj->tuples[k]->mu);
    traj->offPol_weight[k] = policies.back().sampImpWeight; //(race condition)
    //Racer off-pol correction weight: /*lambda*/
    impW *= gamma * std::min((Real)1, policies[k-samp].sampImpWeight);
    if (impW < 1e-3) { //then the imp weight is too small to continue
      lastTPolicy = k-1; //we initialize value of Q_RET to V(state_{k})
      truncated = true; //by acting as if sequence is truncated
      break;
    }
  }

  if(thrID==1)  profiler->stop_start("ADV");
  Real Q_RET = 0;
  if(truncated)
  { //initialize Q_RET to value of state after last off policy correction
    const vector<Real> OT = F[0]->forward(traj, lastTPolicy+1, thrID);
    Q_RET = OT[net_indices[0]];
  }

  for (Uint k=lastTPolicy; k>samp; k--) { //propagate Q to k=0
   const vector<Real> out_k = F[0]->get(traj,k,thrID); // precomputed
   offPolCorrUpdate(traj, k, Q_RET, out_k, policies[k-samp]);
  }

  if(thrID==1)  profiler->stop_start("CMP");
  const Policy_t& pol = policies[0];
  vector<Real> grad=compute(traj,samp, Q_RET, out_cur,pol, polTgt,thrID);
  //printf("gradient: %s\n", print(grad).c_str()); fflush(0);

  if(thrID==1)  profiler->stop_start("BCK");
  F[0]->backward(grad, samp, thrID);
  F[0]->gradient(thrID);
  if(thrID==1)  profiler->stop_start("SLP");
}


inline void offPolCorrUpdate(Sequence*const traj, const Uint samp, Real&Q_RET,
  const vector<Real> output, const Policy_t& pol) const
{
  Q_RET = data->standardized_reward(traj, samp+1) + gamma*Q_RET;
  //Used as target: target policy, target value
  const Advantage_t adv_cur = prepare_advantage(output, &pol);
  const Action_t& act = pol.sampAct; //off policy stored action
  const Real V_hat = output[VsID], A_hat = adv_cur.computeAdvantage(act);
  //prepare rolled Q with off policy corrections for next step:
  Q_RET = std::min((Real)1,pol.sampImpWeight)*(Q_RET -A_hat-V_hat) +V_hat;
  traj->SquaredError[samp] = std::min(pol.sampInvWeight, pol.sampImpWeight); //*std::fabs(Q_RET-A_hat-V_hat);
}

#endif
