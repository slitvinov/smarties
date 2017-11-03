/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once

//#define simpleSigma
//#define BONE
//#define UNBW
//#define UNBR
//#define REALBND (Real)1
#define REALBND CmaxPol
//#define ExpTrust

template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_utils
{
 protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Uint nL = Advantage_t::compute_nL(&aInfo);
  const Real CmaxRet, CmaxPol, DKL_target_orig;
  const Real goalSkipRatio = 0.25/CmaxPol;
  Real DKL_target = DKL_target_orig;
  const vector<Uint> net_outputs, net_indices;
  const vector<Uint> pol_start, adv_start;
  std::vector<std::mt19937>& generators;
  const Uint VsValID = net_indices[0];
  const Uint PenalID = net_indices.back(), QPrecID = net_indices.back()+1;
  const bool bGeometric = CmaxRet>1.1 && nA>1;
  mutable Uint nSkipped = 0, nTried = 0;
  const Real learnRate;
  Real skippedPenal = 1;
  mutable vector<long double> cntValGrad;
  mutable vector<vector<long double>> avgValGrad, stdValGrad;
  Real nStoredSeqs_last = 0;
  vector<Grads*> Kgrad;
  vector<Grads*> Ggrad;
  //#ifdef FEAT_CONTROL
  //  const ContinuousSignControl* task;
  //#endif

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
    //this should go to gamma rather quick:
    const Uint ndata = data->Set[seq]->tuples.size();
    net->prepForBackProp(series_1[thrID], ndata-1);
    net->prepForFwdProp( series_2[thrID], ndata);
    vector<Activation*>& series_cur = *(series_1[thrID]);
    vector<Activation*>& series_hat = *(series_2[thrID]);

    if(thrID==1) profiler->stop_start("FWD");

    for (Uint k=0; k<ndata-1; k++) {
      const Tuple * const _t = data->Set[seq]->tuples[k]; // s, a, mu
      const vector<Real> scaledSold = data->standardize(_t->s);
      //const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
      net->seqPredict_inputs(scaledSold, series_cur[k]);
      #ifdef ACER_PENALIZED
        net->seqPredict_inputs(scaledSold, series_hat[k]);
      #endif
    }
    net->seqPredict_execute(series_cur, series_cur);
    #ifdef ACER_PENALIZED
      net->seqPredict_execute(series_cur, series_hat,
        net->tgt_weights, net->tgt_biases);
    #endif

    if(thrID==1)  profiler->stop_start("CMP");

    Real Q_RET = 0, Q_OPC = 0;
    //if partial sequence then compute value of last state (!= R_end)
    if(not data->Set[seq]->ended)
    {
      const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
      vector<Real> OT(nOutputs, 0), ST =data->standardize(_t->s); //last state
      net->predict(ST, OT, series_cur[ndata-2], series_hat[ndata-1],
          net->tgt_weights, net->tgt_biases);
      Q_OPC = Q_RET = OT[net_indices[0]]; //V(s_T) computed with tgt weights
    }

    for (int k=static_cast<int>(ndata)-2; k>=0; k--)
    {
      const vector<Real> out_cur = net->getOutputs(series_cur[k]);
      Policy_t pol = prepare_policy(out_cur);
      #ifdef ACER_PENALIZED
        const vector<Real> out_hat = net->getOutputs(series_hat[k]);
        Policy_t pol_tgt = prepare_policy(out_hat);
        const Policy_t* const pPol_tgt = &pol_target;
      #else
        const Policy_t* const pPol_tgt = nullptr;
      #endif

      Tuple * const _t = data->Set[seq]->tuples[k];
      pol.prepare(_t->a, _t->mu, bGeometric, pPol_tgt);
      _t->offPol_weight = std::max(pol.sampRhoWeight, pol.sampInvWeight);

      vector<Real>grad=compute(seq,k, Q_RET,Q_OPC, out_cur,pol,pPol_tgt, thrID);
      //#ifdef FEAT_CONTROL
      //const vector<Real> act=aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
      //task->Train(series_cur[k], series_hat[k+1], act, seq, k, grad);
      //#endif

      //write gradient onto output layer:
      net->setOutputDeltas(grad, series_cur[k]);
    }
    #ifndef ExpTrust
      abort(); //TODO
    #endif
    if(thrID==1)  profiler->stop_start("BCK");
    if (thrID==0) net->backProp(series_cur, ndata-1, net->grad);
    else net->backProp(series_cur, ndata-1, net->Vgrad[thrID]);
    /*
    vector<Real> trust = grad_kldiv(seq, samp, policies[0]);
    net->prepForBackProp(series_1[thrID], ndata-1);
    net->setOutputDeltas(trust, series_cur[nRecurr-1]);
    net->backProp(series_cur, nRecurr, Kgrad[thrID]);
    */
    if(thrID==1)  profiler->stop_start("SLP");
  }

  void Train(const Uint seq, const Uint samp, const Uint thrID) const override
  {
    //Code to figure out workload:
    const Uint ndata=data->Set[seq]->tuples.size(), bEnd=data->Set[seq]->ended;
    assert(samp<ndata-1);
    const Uint nMaxTargets = MAX_UNROLL_AFTER+1, nMaxBPTT = MAX_UNROLL_BFORE;
    //for off policy correction we need reward, therefore not last one:
    Uint nSUnroll = min(                     ndata-samp-1, nMaxTargets-1);
    //if we do not have a terminal reward, then we compute value of last state:
    Uint nSValues = min(bEnd? ndata-samp-1 : ndata-samp  , nMaxTargets  );
    //if truncated seq, we cannot compute the OFFPOL correction for the last one
    const Uint nRecurr = bRecurrent ? min(nMaxBPTT,samp)+1        : 1;
    const Uint iRecurr = bRecurrent ? max(nMaxBPTT,samp)-nMaxBPTT : samp;
    //if(thrID==1) { printf("%d %u %u %u %u %u %u\n", bEnd, samp, ndata, nSUnroll, nSValues, nRecurr, iRecurr); fflush(0); }
    if(thrID==1) profiler->stop_start("FWD");

    //allocate stuff
    vector<Real> out_cur(nOutputs,0), out_hat(nOutputs,0);
    net->prepForBackProp(series_1[thrID], nRecurr);
    net->prepForFwdProp(series_2[thrID], nSValues);
    vector<Activation*>& series_cur = *(series_1[thrID]);
    vector<Activation*>& series_hat = *(series_2[thrID]);
    vector<Policy_t> policies;
    policies.reserve(nSValues);

    //propagation of RNN signals:
    for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
      const vector<Real> inp = data->standardize(data->Set[seq]->tuples[k]->s);
      net->seqPredict_inputs(inp, series_cur[j]);
      if(k==samp) { //all are loaded: execute the whole loop:
        assert(j==nRecurr-1);
        net->seqPredict_execute(series_cur, series_cur);
        //extract the only output we actually correct:
        net->seqPredict_output(out_cur, series_cur[j]); //the humanity!
        policies.push_back(prepare_policy(out_cur));

        #ifdef ACER_PENALIZED
          //predict samp with tgt w using curr recurrent inputs as estimate:
          const Activation*const recur = j ? series_cur[j-1] : nullptr;
          net->predict(inp, out_hat, recur, series_hat[0], net->tgt_weights, net->tgt_biases);
        #endif
      }
    }

    #ifdef ACER_PENALIZED
      const Policy_t pol_target = prepare_policy(out_hat);
      const Policy_t* const pPol_tgt = &pol_target;
    #else
      const Policy_t* const pPol_tgt = nullptr;
    #endif

    Tuple * const t0 = data->Set[seq]->tuples[samp];
    policies[0].prepare(t0->a, t0->mu, bGeometric, pPol_tgt);
    const Real rho_cur = policies[0].sampRhoWeight;
    const Real rho_inv = policies[0].sampInvWeight;
    t0->offPol_weight = std::max(rho_inv, rho_cur);

    #if 1
      #pragma omp atomic
      nTried++;

      const Real minImpWeight = std::min((Real)0.5, 1./CmaxPol);
      //const Real maxImpWeight = 10000;
      const Real maxImpWeight = std::max((Real)2.0,    CmaxPol);
      if( rho_cur < minImpWeight || rho_cur > maxImpWeight )
      {
        int newSample = -1;
        #pragma omp critical
        {
          newSample = data->sample(thrID);
          if(newSample >= 0) nSkipped++; //do it inside critical
        }

        if(newSample >= 0) { // process the other sample
          Uint sequence, transition;
          data->indexToSample(newSample, sequence, transition);
          return Train(sequence, transition, thrID);
        } else return;
      }
    #endif

    //compute network for off-policy corrections:
    #ifdef UNBR
      Real impW = policies[0].sampRhoWeight;
    #else
      Real impW = std::min(REALBND, policies[0].sampRhoWeight);
    #endif


    for(Uint k=1; k<nSValues; k++) {
      vector<Real> out_tmp(nOutputs,0);
      const Activation*const recur = k==1 ? series_cur.back() : series_hat[k-1];
      net->predict(data->standardized(seq,k+samp),out_tmp, recur,series_hat[k]);
      policies.push_back(prepare_policy(out_tmp));
      assert(policies.size() == k+1);

      Tuple* const _t = data->Set[seq]->tuples[k+samp];
      policies[k].prepare(_t->a, _t->mu, bGeometric);
      //race condition:
      _t->offPol_weight=max(policies[k].sampInvWeight,policies[k].sampRhoWeight);

      if (k == nSValues-1 && nSValues not_eq nSUnroll) break;

      #ifdef UNBW
        impW *= ACER_LAMBDA*gamma*policies[k].sampImpWeight;
      #else
        impW *= ACER_LAMBDA*gamma*std::min((Real)1,policies[k].sampImpWeight);
      #endif


      if (impW < 1e-6) { //then the imp weight is too small to continue
        //printf("Cut after %u / %u samples!\n",k,nSValues); fflush(stdout);
        nSUnroll = k; //for last state we do not compute offpol correction
        nSValues = k+1; //we initialize value of Q_RET to V(state)
        break;
      }
    }

    if(thrID==1)  profiler->stop_start("ADV");
    Real Q_RET = 0, Q_OPC = 0;
    if(nSValues != nSUnroll) { //partial sequence: compute value of term state
      assert(nSUnroll+1 == nSValues);
      const vector<Real> last_out = net->getOutputs(series_hat[nSValues-1]);
      Q_RET = Q_OPC = last_out[net_indices[0]]; //V(s_T) with tgt weights
    }

    for (int k=static_cast<int>(nSUnroll)-1; k>0; k--) //propagate Q to k=0
     offPolCorrUpdate(seq,k+samp, Q_RET,Q_OPC, net->getOutputs(series_hat[k]), policies[k]);

    if(thrID==1)  profiler->stop_start("CMP");
    vector<Real> grad=compute(seq,samp, Q_RET,Q_OPC, out_cur, policies[0], pPol_tgt, thrID);
    //printf("gradient: %s\n", print(grad).c_str()); fflush(0);

    //#ifdef FEAT_CONTROL
    // const vector<Real> act=aInfo.getInvScaled(data->Set[seq]->tuples[samp]->a);
    // const Activation*const recur = nSValues>1 ? series_hat[1] : nullptr;
    // task->Train(series_cur.back(), recur, act, seq, samp, grad);
    //#endif

    if(thrID==1)  profiler->stop_start("BCK");
      net->setOutputDeltas(grad, series_cur[nRecurr-1]);
      #ifdef ExpTrust //then trust region is computed on batch
        if (thrID==0) net->backProp(series_cur, net->grad);
        else net->backProp(series_cur, nRecurr, net->Vgrad[thrID]);
      #else           //else trust region on this temp gradient
        net->backProp(series_cur, nRecurr, Ggrad[thrID]);
      #endif
    if(thrID==1)  profiler->stop_start("SLP");

    #if 1
      net->prepForBackProp(series_1[thrID], nRecurr);
      vector<Real> trust = grad_kldiv(seq, samp, policies[0]);

      if(thrID==1)  profiler->stop_start("BCK");
        net->setOutputDeltas(trust, series_cur[nRecurr-1]);
        net->backProp(series_cur, nRecurr, Kgrad[thrID]);
      if(thrID==1)  profiler->stop_start("SLP");

      #ifndef ExpTrust
        if (thrID==0) circle_region(Ggrad[thrID], Kgrad[thrID], net->grad, DKL_target);
        else circle_region(Ggrad[thrID], Kgrad[thrID], net->Vgrad[thrID], DKL_target);
        //if (thrID==0) fullstats(Ggrad[thrID], Kgrad[thrID], net->grad, DKL_target);
        //else fullstats(Ggrad[thrID], Kgrad[thrID], net->Vgrad[thrID], DKL_target);
      #endif
    #endif
  }

  inline vector<Real> compute(const Uint seq, const Uint samp, Real& Q_RET,
    Real& Q_OPC, const vector<Real>& out_cur, const Policy_t& pol_cur, const Policy_t* const pol_hat, const Uint thrID) const
  {
    Real meanPena = 0, meanBeta = 0, meanGrad = 0;
    Tuple * const _t = data->Set[seq]->tuples[samp];
    const Real reward = data->standardized_reward(seq, samp+1);
    Q_RET = reward + gamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = reward + gamma*Q_OPC;

    const Real V_cur = out_cur[VsValID], Qprecision = out_cur[QPrecID];
    const Advantage_t adv_cur = prepare_advantage(out_cur, &pol_cur);
    const Action_t& act = pol_cur.sampAct; //unbounded action space

    #ifndef NDEBUG
      adv_cur.test(act, &generators[thrID]);
      pol_cur.test(act, pol_hat);
    #endif

    const Real rho_cur = pol_cur.sampRhoWeight, rho_inv = pol_cur.sampInvWeight;
    //const Real maxImp = std::max((Real)1,rho_cur), oneImp=std::min((Real)1,rho_cur);
    const Real clipImp = std::min(REALBND, rho_cur);
    const Real A_cur = adv_cur.computeAdvantage(act);
    const Real A_OPC = Q_OPC - V_cur;

    //compute quantities needed for trunc import sampl with bias correction
    #if   defined(ACER_TABC)
      const Action_t pol = pol_cur.sample(&generators[thrID]);
      const Real polProbOnPolicy = pol_cur.evalLogProbability(pol);
      const Real polProbBehavior = Policy_t::evalBehavior(pol, _t->mu);
      const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
      const Real A_pol = adv_cur.computeAdvantage(pol);
      const Real gain1 = A_OPC*std::min((Real) 5, rho_cur);
      const Real gain2 = A_pol*std::max((Real) 0, 1-5/rho_pol);

      const vector<Real> gradAcer_1 = pol_cur.policy_grad(act, gain1);
      const vector<Real> gradAcer_2 = pol_cur.policy_grad(pol, gain2);
      const vector<Real> gradAcer = sum2Grads(gradAcer_1, gradAcer_2);
    #else
      #ifdef UNBR
        const Real gain1 = A_OPC*rho_cur;
      #else
        const Real gain1 = A_OPC>0? clipImp*A_OPC : A_OPC*rho_cur;
        //const Real gain1 = clipImp*A_OPC;
      #endif

      const Real DKLmul2 = - A_OPC * rho_inv;
      const vector<Real> gradRacer_1 = pol_cur.policy_grad(act, gain1);
      for(Uint i=0; i<nA; i++) meanGrad += std::fabs(gradRacer_1[1+i]);
      #if 0
        const vector<Real>& gradAcer = gradRacer_1;
        meanBeta = - DKLmul2; //to see it
      #else
        const vector<Real> gradRacer_2 = pol_cur.div_kl_opp_grad(_t->mu, DKLmul2);
        const vector<Real> gradAcer = sum2Grads(gradRacer_1, gradRacer_2);
        for(Uint i=0; i<nA; i++) meanBeta += std::fabs(gradRacer_2[1+i]);
      #endif
    #endif

    #ifdef ACER_PENALIZER
      const Real anneal = iter()>epsAnneal ? 1 : Real(iter())/epsAnneal;
      const Real varCritic = adv_cur.advantageVariance();
      const Real iEpsA = std::pow(A_OPC-A_cur,2)/(varCritic+2.2e-16);
      const Real eta = anneal * safeExp( -0.5*iEpsA);

      const vector<Real> gradC = pol_cur.control_grad(&adv_cur, eta);
      const vector<Real> policy_grad = sum2Grads(gradAcer, gradC);
    #else
      const vector<Real>& policy_grad = gradAcer;
    #endif

    //const Real Q_dist = Q_RET -adv_cur.computeAdvantageNoncentral(act)-V_cur;
    const Real Q_dist = Q_RET -A_cur-V_cur;
    const Real Ver = Q_dist * std::min((Real)1,rho_cur) * std::min((Real)1, Qprecision);
    //const Real Ver = Q_dist * clipImp;
    vector<Real> gradient(nOutputs,0);
    gradient[VsValID] = Ver;
    adv_cur.grad(act, Ver, gradient);
    //decrease prec if err is large, from d Dkl(Q^RET || Q_th)/dQprec
    gradient[QPrecID] = -.5*(Q_dist*Q_dist - 1/Qprecision);
    //gradient[QPrecID] *= gradient[QPrecID] > 0 ? oneImp : maxImp;

    #if defined(ACER_PENALIZED)
      const Real DivKL = pol_cur.kl_divergence_opp(&pol_hat);
      const Real penalDKL = out_cur[PenalID];
      const Real DKLmul1 = - penalDKL;
      //increase if DivKL is greater than Target
      //computed as \nabla_{penalDKL} (DivKL - DKL_target)^4
      //with rough approximation that d DivKL/ d penalDKL \propto penalDKL
      //(distance increases if penalty term increases, similar to PPO )
      //gradient[PenalID] = std::pow((DivKL-DKL_target)/DKL_target,3)*penalDKL;
      gradient[PenalID] = (DivKL - DKL_target)*penalDKL/DKL_target;
      //trust region updating
      const vector<Real> penal_grad = pol_cur.div_kl_opp_grad(&pol_hat,DKLmul1);
      const vector<Real> totalPolGrad = sum2Grads(penal_grad, policy_grad);
      for(Uint i=0; i<nA; i++) meanPena += std::fabs(penal_grad[1+i]);
    #elif defined(ACER_ADAPTIVE) //adapt learning rate:
      gradient[PenalID] = -4*std::pow((DivKL-DKL_target)/DKL_target,3)*opt->eta;
      const vector<Real>& totalPolGrad = policy_grad;
      //avoid races, only one thread updates, should be already redundant:
      if (thrID==1) //if thrd is here, surely we are not updating weights
        opt->eta = out_cur[PenalID];
    #elif defined(ACER_CONSTRAINED)
      const Real DivKL = pol_cur.sampRhoWeight; //unused, just to see it
      const vector<Real> gradDivKL = pol_cur.div_kl_opp_grad(_t->mu, 1);
      const vector<Real> totalPolGrad = trust_region_update(policy_grad, gradDivKL, DKL_target);

      for(Uint i=0;i<nA;i++)meanPena+=fabs(totalPolGrad[1+i]-policy_grad[1+i]);
    #else
      const Real DivKL = pol_cur.sampRhoWeight; //unused, just to see it
      if(thrID == 1) {
        const vector<Real> gradDivKL = pol_cur.div_kl_opp_grad(_t->mu, 1);
        Real normG = 0, normT = 0, dot = 0;
        for(Uint i=0;i<gradDivKL.size();i++) {
          normG += policy_grad[i]*policy_grad[i];
          normT += gradDivKL[i]*gradDivKL[i];
          dot += policy_grad[i]*gradDivKL[i];
        }
        ofstream fs;
        fs.open("grads_dist.txt", ios::app);
        fs<<normG<<"\t"<<normT<<"\t"<<dot<<endl;
        fs.close(); fs.flush();
      }
      const vector<Real>& totalPolGrad = policy_grad;
    #endif

    pol_cur.finalize_grad(totalPolGrad, gradient);
    //prepare Q with off policy corrections for next step:
      Q_RET = std::min((Real)1,pol_cur.sampImpWeight)*(Q_RET-A_cur-V_cur) +V_cur;
    #ifdef UNBW
      Q_OPC = pol_cur.sampImpWeight*(Q_OPC-A_cur-V_cur) +V_cur;
    #else
      Q_OPC = std::min((Real)1,pol_cur.sampImpWeight)*(Q_OPC-A_cur-V_cur)+V_cur;
    #endif
    //bookkeeping:
    dumpStats(Vstats[thrID], A_cur+V_cur, Q_dist); //Ver
    //dumpStats(Vstats[thrID], A_cur+V_cur, Ver ); //Ver

    //write gradient onto output layer:
    const vector<Real> info = { DivKL, meanPena, meanBeta, meanGrad};
    statsGrad(avgValGrad[thrID+1],stdValGrad[thrID+1],cntValGrad[thrID+1],info);
    statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], gradient);
    //int clip =
    clip_gradient(gradient, stdGrad[0], seq, samp);
    //if(clip) printf("A:%f Aret:%f rho:%f g1:%f %f\n",// g2:%f %f\n",
    //A_cur, A_OPC, rho_cur, gradRacer_1[1], gradRacer_1[2]//, gradRacer_2[1],  gradRacer_2[2]
    //);
    _t->SquaredError = std::min(rho_inv, rho_cur);//*std::fabs(Q_dist);
    return gradient;
  }

  inline void offPolCorrUpdate(const Uint seq, const Uint samp, Real& Q_RET,
    Real& Q_OPC, const vector<Real> output, const Policy_t& policy) const
  {
    const Real reward = data->standardized_reward(seq, samp+1);
    Q_RET = reward + gamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = reward + gamma*Q_OPC;
    //Used as target: target policy, target value
    const Advantage_t adv_cur = prepare_advantage(output, &policy);
    const Action_t& act = policy.sampAct; //off policy stored action
    const Real V_hat = output[VsValID], A_hat = adv_cur.computeAdvantage(act);
    //prepare rolled Q with off policy corrections for next step:
      Q_RET = std::min((Real)1,policy.sampImpWeight)*(Q_RET-A_hat-V_hat) +V_hat;
    #ifdef UNBW
      Q_OPC = policy.sampImpWeight*(Q_OPC-A_hat-V_hat) +V_hat;
    #else
      Q_OPC = std::min((Real)1,policy.sampImpWeight)*(Q_OPC-A_hat-V_hat) +V_hat;
    #endif
    data->Set[seq]->tuples[samp]->SquaredError =
      std::min(policy.sampInvWeight,policy.sampRhoWeight); //*std::fabs(Q_RET-A_hat-V_hat);
  }

  inline vector<Real> grad_kldiv(const Uint seq, const Uint samp, const Policy_t& pol_cur) const
  {
    const Tuple * const _t = data->Set[seq]->tuples[samp];
    const vector<Real> gradDivKL = pol_cur.div_kl_opp_grad(_t->mu, 1);
    vector<Real> gradient(nOutputs,0);
    pol_cur.finalize_grad(gradDivKL, gradient);
    //clip_gradient(gradient, stdGrad[0], seq, samp);
    return gradient;
  }

 public:
  RACER(MPI_Comm comm, Environment*const _env, Settings& sett,
    vector<Uint> net_outs, vector<Uint> pol_inds, vector<Uint> adv_inds) :
    Learner_utils(comm, _env, sett, sett.nnOutputs), CmaxRet(sett.opcWeight),
    CmaxPol(sett.impWeight), DKL_target_orig(sett.klDivConstraint),
    net_outputs(net_outs), net_indices(count_indices(net_outs)),
    pol_start(pol_inds), adv_start(adv_inds), generators(sett.generators),
    learnRate(sett.learnrate), cntValGrad(nThreads+1,0),
    avgValGrad(nThreads+1,vector<long double>(4, 0)),
    stdValGrad(nThreads+1,vector<long double>(4, 0))
  {
    //#ifdef FEAT_CONTROL
    //  const Uint task_out0 = ContinuousSignControl::addRequestedLayers(nA,
    //    env->sI.dimUsed, net_indices, net_outputs, out_weight_inits);
    // task = new ContinuousSignControl(task_out0, nA, env->sI.dimUsed, net,data);
    //#endif
    //test();
    if(sett.maxTotSeqNum < sett.batchSize)
    die("maxTotSeqNum < batchSize")
  }
  ~RACER()
  {
    for (auto & trash : Kgrad) _dispose_object(trash);
  }

  void select(const int agentId, const Agent& agent) override
  {
    if(agent.Status==2) { //no need for action, just pass terminal s & r
      data->passData(agentId,agent,vector<Real>(policyVecDim,0));
      return;
    }

    vector<Real> output = output_stochastic_policy(agentId, agent);
    assert(output.size() == nOutputs);
    //variance is pos def: transform linear output layer with softplus

    const Policy_t pol = prepare_policy(output);
    const Advantage_t adv = prepare_advantage(output, &pol);
    const Real anneal = annealingFactor();
    vector<Real> beta = pol.getBeta();
    //if(bTrain) pol.anneal_beta(beta, anneal*greedyEps);

    const Action_t act = pol.finalize(positive(greedyEps+anneal), gen, beta);
    agent.a->set(act);

    #ifdef DUMP_EXTRA
    //beta.insert(beta.end(), adv.matrix.begin(), adv.matrix.end());
    //beta.insert(beta.end(), adv.mean.begin(),   adv.mean.end());
    #else
    //beta.push_back(output[QPrecID]); beta.push_back(output[PenalID]);
    #endif
    data->passData(agentId, agent, beta);
    dumpNetworkInfo(agentId);
  }

  //void test();
  void processStats() override
  {
    {
      stats.minQ= 1e9; stats.MSE =0; stats.dCnt=0;
      stats.maxQ=-1e9; stats.avgQ=0; stats.relE=0;
      for (Uint i=0; i<Vstats.size(); i++) {
        stats.MSE  += Vstats[i]->MSE;  stats.avgQ += Vstats[i]->avgQ;
        stats.stdQ += Vstats[i]->stdQ; stats.dCnt += Vstats[i]->dCnt;
        stats.minQ = std::min(stats.minQ, Vstats[i]->minQ);
        stats.maxQ = std::max(stats.maxQ, Vstats[i]->maxQ);
        Vstats[i]->minQ= 1e9; Vstats[i]->MSE =0; Vstats[i]->dCnt=0;
        Vstats[i]->maxQ=-1e9; Vstats[i]->avgQ=0; Vstats[i]->stdQ=0;
      }

      if (learn_size > 1) {
        MPI_Allreduce(MPI_IN_PLACE,&stats.MSE, 1,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
        MPI_Allreduce(MPI_IN_PLACE,&stats.dCnt,1,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
        MPI_Allreduce(MPI_IN_PLACE,&stats.avgQ,1,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
        MPI_Allreduce(MPI_IN_PLACE,&stats.stdQ,1,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
        MPI_Allreduce(MPI_IN_PLACE,&stats.minQ,1,MPI_LONG_DOUBLE,MPI_MIN,mastersComm);
        MPI_Allreduce(MPI_IN_PLACE,&stats.maxQ,1,MPI_LONG_DOUBLE,MPI_MAX,mastersComm);
      }

      stats.epochCount++;
      const long double sum=stats.avgQ, sumsq=stats.stdQ, cnt=stats.dCnt;
      stats.MSE   = std::sqrt(stats.MSE/cnt); stats.avgQ /= cnt;
      stats.stdQ  = std::sqrt((sumsq-sum*sum/cnt)/cnt);
    }

    processGrads();
    statsVector(avgValGrad, stdValGrad, cntValGrad);
    printf("Policy gains: %s (%s)\n", print(avgValGrad[0]).c_str(),
      print(stdValGrad[0]).c_str()); fflush(0);
    { //shift data / gradient counters to maintain grad stepping to sample
      // collection ratio prescirbed by obsPerStep
      //const Uint nData = read_nData();// nData_0 = nData_b4Train();
      //const Real dataCounter = (Real)nData - (Real)nData_last;
      const Real stepCounter = (Real)opt->nepoch - (Real)nStep_last;
      nData_last += stepCounter*obsPerStep/learn_size;
      nStep_last = opt->nepoch;
    }
    { //update sequences
      assert(nStoredSeqs_last <= data->nSequences); //before pruining
      //const Real mul = (Real)nSequences4Train()/(Real)data->nSequences;
      //data->prune(goalSkipRatio*mul, CmaxPol);
      data->prune(goalSkipRatio, CmaxPol);
      const Real currSeqs = data->nSequences; //after pruning
      opt->eta = (Real)data->nSequences/(Real)nSequences4Train()*learnRate;
      //if(currSeqs >= nSequences4Train())     DKL_target = 1.01*DKL_target;
      //else if(currSeqs < nSequences4Train()) DKL_target = 0.95*DKL_target;
      if(currSeqs > nSequences4Train()) DKL_target = 0.1 + DKL_target;
      else if(currSeqs < nSequences4Train()) DKL_target = DKL_target*0.9;
      nStoredSeqs_last = currSeqs; //after pruning
    }
    printf("nData_last:%lu nData:%u nData_b4Updates:%u Set:%u\n", nData_last,
      read_nData(), nData_b4PolUpdates, data->nSequences); fflush(0);
    const Real ratio = nSkipped/(nTried+nnEPS);
    nSkipped = nTried = 0;

    if(learn_rank) return;

    const Real Qprecision=std::exp(net->biases[net->layers.back()->n1stBias+1]);
    const Real penalDKL=std::exp(net->biases[net->layers.back()->n1stBias]);
    long double sumWeights = 0, distTarget = 0, sumWeightsSq = 0;
    #pragma omp parallel for reduction(+:sumWeights,distTarget,sumWeightsSq)
    for (Uint w=0; w<net->getnWeights(); w++) {
      sumWeights += std::fabs(net->weights[w]);
      sumWeightsSq += net->weights[w]*net->weights[w];
      distTarget += std::fabs(net->weights[w]-net->tgt_weights[w]);
    }

    printf("%lu, rmse:%.2Lg, avg_Q:%.2Lg, stdQ:%.2Lg, minQ:%.2Lg, maxQ:%.2Lg, "
      "weight:[%.0Lg %.0Lg %.2Lg], Qprec:%.3f, penalDKL:%f, rewPrec:%f "
      "skip:[ratio:%g DKLtgt:%g eta:%f]\n",
      opt->nepoch, stats.MSE, stats.avgQ, stats.stdQ, stats.minQ, stats.maxQ,
      sumWeights, sumWeightsSq, distTarget, Qprecision, penalDKL,
      data->invstd_reward, ratio, DKL_target, opt->eta);
      fflush(0);

    ofstream fs;
    fs.open("stats.txt", ios::app);
    fs<<opt->nepoch<<"\t"<<stats.MSE<<"\t"<<stats.avgQ<<"\t"<<stats.stdQ<<"\t"<<
      stats.minQ<<"\t"<<stats.maxQ<<"\t"<<sumWeights<<"\t"<<sumWeightsSq<<"\t"<<
      distTarget<<"\t"<<Qprecision<<"\t"<<penalDKL<<"\t"<<data->invstd_reward<<
      "\t"<<ratio<<"\t"<<DKL_target<<endl;
      fs.close(); fs.flush();
    if (stats.epochCount % 100==0) save("policy");
  }

  void applyGradient() override
  {
    if(!nAddedGradients) {
      nData_last = 0;
      nStoredSeqs_last = data->nSequences;
      assert(data->Set.size() == data->nSequences);
      return;
    }

    assert(nSkipped<nTried || nTried == 0);
    //printf("%g %u %u %g %u %g\n", ratio, nSkipped, nTried, skippedPenal, data->adapt_TotSeqNum, opt->eta); fflush(0);
    Learner::applyGradient();
  }

  #ifdef ExpTrust
  void stackAndUpdateNNWeights() override
  {
    if(!nAddedGradients) die("Error in stackAndUpdateNNWeights\n");
    assert(bTrain);
    opt->nepoch++;
    Uint nTotGrads = nAddedGradients;
    opt->stackGrads(Kgrad[0], Kgrad);
    opt->stackGrads(net->grad, net->Vgrad); //add up gradients across threads
    if (learn_size > 1) { //add up gradients across masters
      MPI_Allreduce(MPI_IN_PLACE, Kgrad[0]->_W, net->getnWeights(),
          MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, Kgrad[0]->_B, net->getnBiases(),
          MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, net->grad->_W, net->getnWeights(),
          MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, net->grad->_B, net->getnBiases(),
          MPI_NNVALUE_TYPE, MPI_SUM, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE,&nTotGrads,1,MPI_UNSIGNED,MPI_SUM,mastersComm);
    }

    circle_region(Kgrad[0], net->grad, DKL_target, nTotGrads);
    //update is deterministic: can be handled independently by each node
    //communication overhead is probably greater than a parallelised sum
    opt->update(net->grad, nTotGrads);
  }
  #endif
};
