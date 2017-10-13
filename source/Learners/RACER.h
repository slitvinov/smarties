/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Learner_utils.h"
#include "../Math/FeatureControlTasks.h"
#include "../Math/Quadratic_advantage.h"

#ifndef ADV_QUAD
#include "../Math/Mixture_advantage_prova.h"
#warning "Using Mixture_advantage with Gaussian advantages"
#else
#include "../Math/Mixture_advantage.h"
#warning "Using Mixture_advantage with Quadratic advantages"
#endif

//template<Uint NEXPERTS> //does not work, my life is a lie!
#ifndef NEXPERTS
#define NEXPERTS 1
#warning "Using Mixture_advantage with 1 expert"
#endif

#include "../Math/Discrete_policy.h"
//#define simpleSigma

template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_utils
{
 protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Uint nL = Advantage_t::compute_nL(&aInfo);
  const Real CmaxRet, CmaxPol, goalSkipRatio = 0.05;
  Real DKL_target;
  const vector<Uint> net_outputs, net_indices;
  const vector<Uint> pol_start, adv_start;
  std::vector<std::mt19937>& generators;
  const Uint VsValID = net_indices[0];
  const Uint PenalID = net_indices.back(), QPrecID = net_indices.back()+1;
  const bool bGeometric = CmaxRet>1 && nA>1;
  mutable Uint nSkipped = 0, nTried = 0;
  const Real learnRate;
  Real skippedPenal = 1;
  mutable vector<long double> cntValGrad;
  mutable vector<vector<long double>> avgValGrad, stdValGrad;
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
    net->prepForFwdProp( series_2[thrID], ndata-1);
    vector<Activation*>& series_cur = *(series_1[thrID]);
    vector<Activation*>& series_hat = *(series_2[thrID]);

    if(thrID==1) profiler->stop_start("FWD");

    for (Uint k=0; k<ndata-1; k++) {
      const Tuple * const _t = data->Set[seq]->tuples[k]; // s, a, mu
      const vector<Real> scaledSold = data->standardize(_t->s);
      //const vector<Real> scaledSold = data->standardize(_t->s, 0.01, thrID);
      net->seqPredict_inputs(scaledSold, series_cur[k]);
      net->seqPredict_inputs(scaledSold, series_hat[k]);
    }
    net->seqPredict_execute(series_cur, series_cur);
    net->seqPredict_execute(series_cur, series_hat,
      net->tgt_weights, net->tgt_biases);

    if(thrID==1)  profiler->stop_start("CMP");

    Real Q_RET = 0, Q_OPC = 0;
    //if partial sequence then compute value of last state (!= R_end)
    if(not data->Set[seq]->ended) {
      series_hat.push_back(net->allocateActivation());
      const Tuple * const _t = data->Set[seq]->tuples[ndata-1];
      vector<Real> OT(nOutputs, 0), ST =data->standardize(_t->s); //last state
      net->predict(ST,OT, series_hat,ndata-1,net->tgt_weights,net->tgt_biases);
      Q_OPC = Q_RET = OT[net_indices[0]]; //V(s_T) computed with tgt weights
    }

    for (int k=static_cast<int>(ndata)-2; k>=0; k--)
    {
      const vector<Real> out_cur = net->getOutputs(series_cur[k]);
      const vector<Real> out_hat = net->getOutputs(series_hat[k]);
      Policy_t pol_cur = prepare_policy(out_cur);
      Policy_t pol_tgt = prepare_policy(out_hat);
      const Tuple * const _t = data->Set[seq]->tuples[k];
      pol_cur.prepare(_t->a, _t->mu, bGeometric, &pol_tgt);
      vector<Real>grad=compute(seq,k,Q_RET,Q_OPC,out_cur,pol_cur,pol_tgt,thrID);
      //#ifdef FEAT_CONTROL
      //const vector<Real> act=aInfo.getInvScaled(data->Set[seq]->tuples[k]->a);
      //task->Train(series_cur[k], series_hat[k+1], act, seq, k, grad);
      //#endif

      //write gradient onto output layer:
      statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
      clip_gradient(grad, stdGrad[0], seq, k);
      net->setOutputDeltas(grad, series_cur[k]);
    }

    if(thrID==1)  profiler->stop_start("BCK");
    if (thrID==0) net->backProp(series_cur, ndata-1, net->grad);
    else net->backProp(series_cur, ndata-1, net->Vgrad[thrID]);
    if(thrID==1)  profiler->stop_start("SLP");
  }

  void Train(const Uint seq, const Uint samp, const Uint thrID) const override
  {
    //Code to figure out workload:
    const Uint ndata = data->Set[seq]->tuples.size();
    assert(samp<ndata-1);
    const bool bEnd = data->Set[seq]->ended; //whether sequence has terminal rew
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
        //predict samp with target weight using curr recurrent inputs as estimate:
        const Activation*const recur = j ? series_cur[j-1] : nullptr;
        net->predict(inp, out_hat, recur, series_hat[0], net->tgt_weights, net->tgt_biases);
      }
    }
    const Policy_t pol_target = prepare_policy(out_hat);
    const Tuple * const t0 = data->Set[seq]->tuples[samp];
    policies[0].prepare(t0->a, t0->mu, bGeometric, &pol_target);

    #pragma omp atomic
    nTried++;

    if(policies[0].sampRhoWeight < std::min((Real)0.2, 1/CmaxPol))
    {
      int newSample = -1;
      #pragma omp critical
      {
        newSample = data->sample(thrID);
        if(newSample >= 0) nSkipped++; //do it inside critical
      }

      if(newSample >= 0) {
        Uint sequence, transition;
        data->indexToSample(newSample, sequence, transition);
        return Train(sequence, transition, thrID);
      }
    }

    //compute network for off-policy corrections:
    Real impW = std::min((Real)1, policies[0].sampRhoWeight);
    for(Uint k=1; k<nSValues; k++)
    {
      vector<Real> out_tmp(nOutputs,0);
      net->predict(data->standardized(seq, k+samp), out_tmp, series_hat, k);
      policies.push_back(prepare_policy(out_tmp));
      assert(policies.size() == k+1);

      #ifndef NO_CUT_TRACES
        if (k == nSValues-1 && nSValues not_eq nSUnroll) break;
        const Tuple* const _t = data->Set[seq]->tuples[k+samp];
        policies[k].prepare(_t->a, _t->mu, bGeometric);
        const Real clipValW = std::min((Real)1, policies[k].sampImpWeight);
        impW *= ACER_LAMBDA*gamma*clipValW;
        if (impW < 1e-3) { //then the imp weight is too small to continue
          //printf("Cut after %u / %u samples!\n", k,nSValues);fflush(stdout);
          nSUnroll = k; //for last state we do not compute offpol correction
          nSValues = k+1; //we initialize value of Q_RET to V(state)
          break;
        }
      #endif
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
    vector<Real> grad=compute(seq,samp, Q_RET,Q_OPC, out_cur, policies[0], pol_target, thrID);
    //printf("gradient: %s\n", print(grad).c_str()); fflush(0);
    //#ifdef FEAT_CONTROL
    // const vector<Real> act=aInfo.getInvScaled(data->Set[seq]->tuples[samp]->a);
    // const Activation*const recur = nSValues>1 ? series_hat[1] : nullptr;
    // task->Train(series_cur.back(), recur, act, seq, samp, grad);
    //#endif

    //write gradient onto output layer:
    statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
    clip_gradient(grad, stdGrad[0], seq, samp);
    net->setOutputDeltas(grad, series_cur.back());

    if(thrID==1)  profiler->stop_start("BCK");
    if (thrID==0) net->backProp(series_cur, net->grad);
    else net->backProp(series_cur, net->Vgrad[thrID]);
    if(thrID==1)  profiler->stop_start("SLP");
  }

  inline vector<Real> compute(const Uint seq, const Uint samp, Real& Q_RET,
    Real& Q_OPC, const vector<Real>& out_cur, const Policy_t& pol_cur, const Policy_t& pol_hat, const Uint thrID) const
  {
    const Tuple * const _t = data->Set[seq]->tuples[samp];
    const Real reward = data->standardized_reward(seq, samp+1);
    Q_RET = reward + gamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = reward + gamma*Q_OPC;
    //get everybody camera ready:
    const Real V_cur = out_cur[VsValID], Qprecision = out_cur[QPrecID];
    const Advantage_t adv_cur = prepare_advantage(out_cur, &pol_cur);
    const Action_t& act = pol_cur.sampAct; //unbounded action space
    #ifndef NDEBUG
      adv_cur.test(act, &generators[thrID]);
      pol_cur.test(act, &pol_hat);
    #endif
    const Real rho_cur = pol_cur.sampRhoWeight, rho_inv = 1/(rho_cur+nnEPS);
    const Real DivKL = pol_cur.kl_divergence_opp(&pol_hat);
    const Real A_cur = adv_cur.computeAdvantage(act);
    const Real A_OPC = Q_OPC - V_cur, Q_dist = Q_RET -A_cur-V_cur;

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
      const Real gain1 = A_OPC>0? min(CmaxPol,rho_cur)*A_OPC : A_OPC*rho_cur;
      const vector<Real> gradAcer = pol_cur.policy_grad(act, gain1);
    #endif

    #ifdef ACER_PENALIZER
      const Real anneal = iter()>epsAnneal ? 1 : Real(iter())/epsAnneal;
      const Real varCritic = adv_cur.advantageVariance();
      const Real iEpsA = std::pow(A_OPC-A_cur,2)/(varCritic+2.2e-16);
      const Real eta = anneal * safeExp( -0.5*iEpsA);

      const vector<Real> gradC = pol_cur.control_grad(&adv_cur, eta);
      const vector<Real> policy_grad = sum2Grads(gradAcer, gradC);
    #else
      const vector<Real> policy_grad = gradAcer;
    #endif

    const Real logEpsilon = std::log( std::numeric_limits<Real>::epsilon() );
    const Real Ver = Q_dist * std::min((Real)1, rho_cur);
    const Real Qer = pol_cur.sampLogPonPolicy > logEpsilon ? Q_dist : 0;
    vector<Real> gradient(nOutputs,0);
    gradient[VsValID]= (Qer+Ver) * Qprecision;

    #if defined(ACER_CONSTRAINED)
      const vector<Real> gradDivKL = pol_cur.div_kl_grad(&pol_hat);
      const vector<Real> totalPolGrad = trust_region_update(policy_grad, gradDivKL, DKL_hardmax);
    #elif defined(ACER_ADAPTIVE) //adapt learning rate:
      gradient[PenalID] = -4*std::pow(DivKL - DKL_target,3)*opt->eta;
      const vector<Real> totalPolGrad = policy_grad;
      //avoid races, only one thread updates, should be already redundant:
      if (thrID==1) //if thrd is here, surely we are not updating weights
        opt->eta = out_cur[PenalID];
    #else
      const Real penalDKL = out_cur[PenalID];
      const Real DKLmul1 = - skippedPenal * penalDKL;
      //increase if DivKL is greater than Target
      //computed as \nabla_{penalDKL} (DivKL - DKL_target)^4
      //with rough approximation that d DivKL/ d penalDKL \propto penalDKL
      //(distance increases if penalty term increases, similar to PPO )
      gradient[PenalID] = 4*std::pow(DivKL - DKL_target,3)*penalDKL;
      //trust region updating
      const vector<Real> penal_grad=pol_cur.div_kl_opp_grad(&pol_hat, DKLmul1);
      const vector<Real> totalPolGrad = sum2Grads(penal_grad, policy_grad);
    #endif

    const Real penalBeta = std::max(A_OPC, (Real)0)*std::min(rho_inv, CmaxPol);
    const Real DKLmul2 = - skippedPenal * penalBeta;
    const vector<Real> beta_grad =pol_cur.div_kl_opp_grad(_t->mu, DKLmul2);
    const vector<Real> finalPolGrad = sum2Grads(totalPolGrad, beta_grad);

    //decrease precision if error is large
    //computed as \nabla_{Qprecision} Dkl (Q^RET_dist || Q_dist)
    gradient[QPrecID] = -.5*(Q_dist*Q_dist - 1/Qprecision);
    //adv_cur.grad(act, Qer, gradient, aInfo.bounded);
    adv_cur.grad(act, Qer * Qprecision, gradient);
    pol_cur.finalize_grad(finalPolGrad, gradient);
    //prepare Q with off policy corrections for next step:
    Q_RET = std::min((Real)1, pol_cur.sampImpWeight)*Q_dist +V_cur;
    Q_OPC = std::min((Real)1, pol_cur.sampImpWeight)*Q_dist +V_cur;
    //bookkeeping:
    dumpStats(Vstats[thrID], A_cur+V_cur, Ver ); //Ver
    //{
    //const vector<Real> info = {std::fabs(gain1), penalBeta, penalDKL};
    //statsGrad(avgValGrad[thrID+1],stdValGrad[thrID+1],cntValGrad[thrID+1],info);
    //}
    data->Set[seq]->tuples[samp]->SquaredError = -rho_inv -rho_cur;
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
    Q_OPC = std::min((Real)1,policy.sampImpWeight)*(Q_RET-A_hat-V_hat) +V_hat;
  }

 public:
  RACER(MPI_Comm comm, Environment*const _env, Settings& sett,
    vector<Uint> net_outs, vector<Uint> pol_inds, vector<Uint> adv_inds) :
    Learner_utils(comm, _env, sett, sett.nnOutputs), CmaxRet(sett.opcWeight),
    CmaxPol(sett.impWeight), DKL_target(sett.klDivConstraint),
    net_outputs(net_outs), net_indices(count_indices(net_outs)),
    pol_start(pol_inds), adv_start(adv_inds), generators(sett.generators),
    learnRate(sett.learnrate), cntValGrad(nThreads+1,0),
    avgValGrad(nThreads+1,vector<long double>(3,0)), stdValGrad(nThreads+1,vector<long double>(3,0))
  {
    //#ifdef FEAT_CONTROL
    //  const Uint task_out0 = ContinuousSignControl::addRequestedLayers(nA,
    //    env->sI.dimUsed, net_indices, net_outputs, out_weight_inits);
    // task = new ContinuousSignControl(task_out0, nA, env->sI.dimUsed, net,data);
    //#endif
    //test();
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
    if(bTrain) pol.anneal_beta(beta, anneal*greedyEps);

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
    stats.minQ= 1e9;stats.MSE =0;stats.dCnt=0;
    stats.maxQ=-1e9;stats.avgQ=0;stats.relE=0;

    for (Uint i=0; i<Vstats.size(); i++) {
      stats.MSE  += Vstats[i]->MSE;
      stats.avgQ += Vstats[i]->avgQ;
      stats.stdQ += Vstats[i]->stdQ;
      stats.dCnt += Vstats[i]->dCnt;
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
    epochCounter = stats.epochCount;
    const long double sum=stats.avgQ, sumsq=stats.stdQ, cnt=stats.dCnt;
    //stats.MSE  /= cnt-1;
    stats.MSE   = std::sqrt(stats.MSE/cnt);
    stats.avgQ /= cnt; //stats.relE/=stats.dCnt;
    stats.stdQ  = std::sqrt((sumsq-sum*sum/cnt)/cnt);
    sumElapsed = 0; countElapsed=0;
    processGrads();
    //statsVector(avgValGrad, stdValGrad, cntValGrad);
    //printf("Policy gains: %s (%s)\n", print(avgValGrad[0]).c_str(),
    //  print(stdValGrad[0]).c_str()); fflush(0);

    if(learn_rank) return;

    long double sumWeights = 0, distTarget = 0, sumWeightsSq = 0;

    #pragma omp parallel for reduction(+:sumWeights,distTarget,sumWeightsSq)
    for (Uint w=0; w<net->getnWeights(); w++) {
      sumWeights += std::fabs(net->weights[w]);
      sumWeightsSq += net->weights[w]*net->weights[w];
      distTarget += std::fabs(net->weights[w]-net->tgt_weights[w]);
    }

    const Real Qprecision = exp(net->biases[net->layers.back()->n1stBias+1]);
    const Real penalDKL = exp(net->biases[net->layers.back()->n1stBias]);

    //shift counters
    const Uint nData = read_nData();// nData_0 = nData_b4Train();
    //const Real dataCounter = nData - (Real)nData_last;
    const Real stepCounter = opt->nepoch - (Real)nStep_last;

    printf("nData_last:%lu nStep_last:%lu nData:%u nStep:%lu\n",
      nData_last, nStep_last, nData, opt->nepoch); fflush(0);
    //nData_last = stepCounter*obsPerStep/learn_size;
    //nStep_last = opt->nepoch;

    const Real ratio = nSkipped/(nTried+nnEPS);
    const Real invratio = nTried/(nSkipped+nnEPS);
    if(ratio>goalSkipRatio) {
      //increase observations per step and reduce buffer size
      obsPerStep = obsPerStep * ratio/goalSkipRatio;
      data->adapt_TotSeqNum = (goalSkipRatio*invratio)*data->adapt_TotSeqNum;
    }
    else {
      obsPerStep = obsPerStep*0.99;
      data->adapt_TotSeqNum = min(data->maxTotSeqNum, data->adapt_TotSeqNum+1);
    }

    //during normal training this should practically have no effect
    //(small perturbation on gradient)
    //added here to make sure that user does not get stuck
    //if ratio>goalSkipRatio DKL_target is reduced, else is increased
    // clipped to 1/1e-3
    DKL_target = clip(DKL_target * goalSkipRatio*invratio, 1, 1e-3);

    printf("%lu, rmse:%.2Lg, avg_Q:%.2Lg, stdQ:%.2Lg, minQ:%.2Lg, maxQ:%.2Lg, "
    "weight:[%.0Lg %.0Lg %.2Lg], Qprec:%.3f, penalDKL:%f, rewPrec:%f "
    "skip:[ratio:%g obsPerStep:%g seqNum:%u DKLtgt:%g penal:%g] %u %lu %lu\n",
      opt->nepoch, stats.MSE, stats.avgQ, stats.stdQ,
      stats.minQ, stats.maxQ, sumWeights, sumWeightsSq, distTarget,
      Qprecision, penalDKL, data->invstd_reward, ratio, obsPerStep,
      data->adapt_TotSeqNum, DKL_target, skippedPenal,
      nData, nData_last, nStep_last);
      fflush(0);
    nSkipped = nTried = 0;
    ofstream fs;
    fs.open("stats.txt", ios::app);
    fs<<opt->nepoch<<"\t"<<stats.MSE<<"\t"<<stats.avgQ<<"\t"<<stats.stdQ<<"\t"<<
    stats.minQ<<"\t"<<stats.maxQ<<"\t"<<sumWeights<<"\t"<<sumWeightsSq<<"\t"<<
    distTarget<<"\t"<<Qprecision<<"\t"<<penalDKL<<"\t"<<data->invstd_reward<<endl;
    fs.close();
    fs.flush();
    if (stats.epochCount % 100==0) save("policy");
  }

  void applyGradient() override
  {
    assert(nSkipped<nTried || nTried == 0);
    const Real ratio = nSkipped/(nTried+nnEPS);
    //skippedPenal goes to 1 if i do not skip any sequecnce
    //goes to inf if i skipp all sequences that i sample
    //this is a safety measure, adaptive mem buffer keeps this number around 1
    //why needed? because if user selects learn rate that is too high
    //then you might skip all samples and code looks stuck
    //because for each skipped sample code computes one network fwd prop
    skippedPenal = std::exp(ratio/(1-ratio));
    opt->eta = learnRate/skippedPenal;

    if(skippedPenal>1.5)
      _warn("Network learn rate is too high, RACER is automatically reducing it to %g. Consider running again with better hyperparameters.\n",opt->eta);

    printf("%g %u %u %g %u %g\n", ratio, nSkipped, nTried, skippedPenal, data->adapt_TotSeqNum, opt->eta); fflush(0);
    Learner::applyGradient();
  }
};

class RACER_cont : public RACER<Quadratic_advantage, Gaussian_policy, vector<Real> >
{
  static vector<Uint> count_outputs(const ActionInfo& aI)
  {
    const Uint nL = Quadratic_term::compute_nL(&aI);
    #if defined ACER_RELAX
      return vector<Uint>{1, nL, aI.dim, aI.dim, 2};
    #else
      return vector<Uint>{1, nL, aI.dim, aI.dim, aI.dim, 2};
    #endif
  }
  static vector<Uint> count_pol_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    #if defined ACER_RELAX
    return vector<Uint>{indices[2], indices[3]};
    #else
    return vector<Uint>{indices[2], indices[4]};
    #endif
  }
  static vector<Uint> count_adv_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    #if defined ACER_RELAX
    return vector<Uint>{indices[1]};
    #else
    return vector<Uint>{indices[1], indices[3]};
    #endif
  }

 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    const Uint nL = Quadratic_advantage::compute_nL(aI);
    #if defined ACER_RELAX // I output V(s), P(s), pol(s), prec(s) (and variate)
      return 1 + nL + aI->dim + aI->dim + 2;
    #else // I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
      return 1 + nL + aI->dim + aI->dim + aI->dim + 2;
    #endif
  }

  RACER_cont(MPI_Comm comm, Environment*const _env, Settings & settings) :
  RACER(comm, _env, settings, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Continuous-action RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    Builder build(settings);

    #if defined ACER_RELAX
      vector<Uint> outs{1, nL, nA};
    #else
      vector<Uint> outs{1, nL, nA, nA};
    #endif
    #ifndef simpleSigma
      outs.push_back(nA);
    #endif

    build.stackSimple( vector<Uint>{nInputs}, outs );

    #ifdef simpleSigma
      build.addParamLayer(nA, "Linear", -2*std::log(greedyEps));
    #endif

    //add klDiv penalty coefficient layer, and stdv of Q distribution
    build.addParamLayer(2, "Exp", 0);

    net = build.build();

    #if defined(ACER_ADAPTIVE)
    //set initial value for klDiv penalty coefficient
    Uint penalparid = net->layers.back()->n1stBias; //(was last added layer)
    net->biases[penalparid] = std::log(settings.learnrate);
    #else
    //set initial value for klDiv penalty coefficient
    const Uint penalparid= net->layers.back()->n1stBias;//(was last added layer)
    net->biases[penalparid] = -std::log(settings.klDivConstraint);
    //*tgtUpdateAlpha/settings.learnrate
    #endif
    //printf("Setting bias %d to %f\n",penalparid,net->biases[penalparid]);


    finalize_network(build);

    #ifdef DUMP_EXTRA
     policyVecDim = 2*nA + nL;
    #else
     policyVecDim = 2*nA;
    #endif
  }
};

class RACER_disc : public RACER<Discrete_advantage, Discrete_policy, Uint>
{
  static vector<Uint> count_outputs(const ActionInfo*const aI)
  {
    return vector<Uint>{1, aI->maxLabel, aI->maxLabel, 2};
  }
  static vector<Uint> count_pol_starts(const ActionInfo*const aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[1]};
  }
  static vector<Uint> count_adv_starts(const ActionInfo*const aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[2]};
  }

 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    return 1 + aI->maxLabel + aI->maxLabel + 2;
  }

 public:
  RACER_disc(MPI_Comm comm, Environment*const _env, Settings & settings) :
  RACER( comm, _env, settings, count_outputs(&_env->aI), count_pol_starts(&_env->aI), count_adv_starts(&_env->aI) )
  {
    printf("Discrete-action RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    Builder build(settings);
    vector<Uint> outs{1, nL, nA};
    build.stackSimple( vector<Uint>{nInputs}, outs );
    //add klDiv penalty coefficient layer, and stdv of Q distribution
    build.addParamLayer(2, "Exp", 0);

    net = build.build();

    #if defined(ACER_ADAPTIVE)
    //set initial value for klDiv penalty coefficient
    Uint penalparid = net->layers.back()->n1stBias; //(was last added layer)
    net->biases[penalparid] = std::log(settings.learnrate);
    #else
    //set initial value for klDiv penalty coefficient
    Uint penalparid = net->layers.back()->n1stBias; //(was last added layer)
    net->biases[penalparid] = -std::log(1);
    #endif
    //printf("Setting bias %d to %f\n",penalparid,net->biases[penalparid]);

    finalize_network(build);

    #ifdef DUMP_EXTRA
     policyVecDim = 2*nA;
    #else
     policyVecDim = nA;
    #endif
  }
};

class RACER_experts : public RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, vector<Real>>
{
  static vector<Uint> count_outputs(const ActionInfo& aI)
  {
    const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(&aI);
    return vector<Uint>{1, nL, NEXPERTS, NEXPERTS*aI.dim, NEXPERTS*aI.dim, 2};
  }
  static vector<Uint> count_pol_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[2], indices[3], indices[4]};
  }
  static vector<Uint> count_adv_starts(const ActionInfo& aI)
  {
    const vector<Uint> sizes = count_outputs(aI);
    const vector<Uint> indices = count_indices(sizes);
    return vector<Uint>{indices[1]};
  }
 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(aI);
    return 1 + nL + NEXPERTS*(1 +2*aI->dim) + 2;
  }

  RACER_experts(MPI_Comm comm, Environment*const _env, Settings & settings) :
  RACER(comm, _env, settings, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Continuous-action RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    Builder build(settings);

    vector<Uint> outs{1, nL, NEXPERTS, NEXPERTS*nA, NEXPERTS*nA};
    build.stackSimple( vector<Uint>{nInputs}, outs );
    //add klDiv penalty coefficient layer, and stdv of Q distribution:
    build.addParamLayer(2, "Exp", 0);
    net = build.build();

    #if defined(ACER_ADAPTIVE)
    //set initial value for klDiv penalty coefficient
    Uint penalparid = net->layers.back()->n1stBias; //(was last added layer)
    net->biases[penalparid] = std::log(settings.learnrate);
    #else
    //set initial value for klDiv penalty coefficient
    const Uint penalparid= net->layers.back()->n1stBias;//(was last added layer)
    net->biases[penalparid] = -std::log(settings.klDivConstraint);
    //*tgtUpdateAlpha/settings.learnrate
    #endif
    //printf("Setting bias %d to %f\n",penalparid,net->biases[penalparid]);

    finalize_network(build);
    policyVecDim = NEXPERTS +2*NEXPERTS*nA;
  }
};
