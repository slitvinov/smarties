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
#include "../Math/Discrete_policy.h"
//#define simpleSigma

template<typename Advantage_t, typename Policy_t, typename Action_t>
class RACER : public Learner_utils
{
 protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Uint nL = Advantage_t::compute_nL(&aInfo);
  const Real CmaxRet, DKL_target;
  vector<Uint> net_outputs, net_indices;
  const vector<Uint> pol_start, adv_start;
  std::vector<std::mt19937>& generators;
  const Uint PenalID = net_indices.back();
  const Uint QPrecID = net_indices.back()+1;
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
      vector<Real> outT(nOutputs, 0), ST = data->standardize(_t->s); //last state
      net->predict(ST,outT, series_hat,ndata-1, net->tgt_weights,net->tgt_biases);
      Q_OPC = Q_RET = outT[net_indices[0]]; //V(s_T) computed with tgt weights
    }

    for (int k=static_cast<int>(ndata)-2; k>=0; k--)
    {
      vector<Real> out_cur = net->getOutputs(series_cur[k]);
      vector<Real> out_hat = net->getOutputs(series_hat[k]);
      vector<Real> grad = compute(seq,k,Q_RET,Q_OPC,out_cur,out_hat,thrID);
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

    if (thrID==0) net->backProp(series_cur, net->grad);
    else net->backProp(series_cur, net->Vgrad[thrID]);

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

    //propagation of RNN signals:
    for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
      const vector<Real> inp = data->standardize(data->Set[seq]->tuples[k]->s);
      net->seqPredict_inputs(inp, series_cur[j]);
      if(k==samp) { //all are loaded: execute the whole loop:
        assert(j==nRecurr-1);
        net->seqPredict_execute(series_cur, series_cur);
        //extract the only output we actually correct:
        net->seqPredict_output(out_cur, series_cur[j]); //the humanity!
        //predict samp with target weight using curr recurrent inputs as estimate:
        const Activation*const recur = j ? series_cur[j-1] : nullptr;
        net->predict(inp, out_hat, recur, series_hat[0], net->tgt_weights, net->tgt_biases);
      }
    }

    //compute network for off-policy corrections:
    Real importanceW = 1;
    for(Uint k=1; k<nSValues; k++)
    {
      vector<Real> out_tmp(nOutputs,0);
      net->predict(data->standardized(seq, k+samp), out_tmp, series_hat, k);
      //net->predict(data->standardized(seq, k+samp), out_tmp, series_hat, k, net->tgt_weights, net->tgt_biases);

      #ifndef NO_CUT_TRACES
        if (k == nSValues-1) break;
        const Tuple* const _t = data->Set[seq]->tuples[k+samp];
        //else check if the importance weight is too small to continue:
        importanceW *= computeImportanceWeight(out_tmp, _t);
        if (importanceW < 1e-2) {
          //printf("Cut trace after %u out of %u samples!\n",k,nSValues);
          //fflush(stdout);
          nSUnroll = k; //for last state we do not compute offpol correction
          nSValues = k+1; //we initialize value of Q_RET to V(state)
          break;
        }
      #endif
    }
    /*
    if(data->Set.size() == 5000) {
      char asciipath[256];
      sprintf(asciipath, "trunc_seqs_%d.txt", thrID);
      ofstream filestats;
      filestats.open(asciipath, ios::app);
      filestats<<data->nSeenSequences-data->Set[seq]->ID<<" "<<nSValues<<" "<<ndata-1-samp<<endl;
      filestats.close();
      filestats.flush();
    }
    */

    if(thrID==1)  profiler->stop_start("ADV");

    Real Q_RET = 0, Q_OPC = 0;
    if(nSValues != nSUnroll) { //partial sequence: compute value of term state
      const vector<Real> last_out = net->getOutputs(series_hat[nSValues-1]);
      Q_RET = Q_OPC = last_out[net_indices[0]]; //V(s_T) with tgt weights
    }

    for (int k=static_cast<int>(nSUnroll)-1; k>0; k--) //propagate Q to k=0
     offPolCorrUpdate(seq,k+samp, Q_RET,Q_OPC, net->getOutputs(series_hat[k]));

    if(thrID==1)  profiler->stop_start("CMP");

    vector<Real> grad=compute(seq,samp,Q_RET,Q_OPC,out_cur,out_hat,thrID);

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
    Real& Q_OPC, const vector<Real>& out_cur, const vector<Real>& out_hat,
    const Uint thrID) const
  {
    const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
    //const Tuple * const t_ =data->Set[seq]->tuples[samp+1]; //contains r, sNew
    const Real reward = data->standardized_reward(seq, samp+1);
    Q_RET = reward + gamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = reward + gamma*Q_OPC;
    //get everybody camera ready:
    const Real V_cur = out_cur[net_indices[0]]; //V_hat = out_hat[net_indices[0]];
    const Real Qprecision = out_cur[QPrecID];
    const Policy_t pol_cur = prepare_policy(out_cur);
    const Policy_t pol_hat = prepare_policy(out_hat);
    const Advantage_t adv_cur = prepare_advantage(out_cur, &pol_cur);
    const Real A_OPC = Q_OPC - V_cur;

    //off policy stored action and on-policy sample:
    const Action_t act = pol_cur.map_action(_t->a); //unbounded action space
    const Real actProbOnPolicy = pol_cur.evalLogProbability(act);
    const Real actProbBehavior = Policy_t::evalBehavior(act,_t->mu);
    const Real rho_cur = min(MAX_IMPW,safeExp(actProbOnPolicy-actProbBehavior));
    const Real DivKL = pol_cur.kl_divergence_opp(&pol_hat);
    const Real A_cur = adv_cur.computeAdvantage(act);
    const Real Qer = Q_RET -A_cur -V_cur;

    //compute quantities needed for trunc import sampl with bias correction
    #if   defined(ACER_TABC)
      const Action_t pol = pol_cur.sample(&generators[thrID]);
      const Real polProbOnPolicy = pol_cur.evalLogProbability(pol);
      const Real polProbBehavior = Policy_t::evalBehavior(pol,_t->mu);
      const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
      const Real A_pol = adv_cur.computeAdvantage(pol);
      const Real gain1 = A_OPC*min(rho_cur, 5.);
      const Real gain2 = A_pol*max(0.,1 - 5./rho_pol);

      const vector<Real> gradAcer_1 = pol_cur.policy_grad(act, gain1);
      const vector<Real> gradAcer_2 = pol_cur.policy_grad(pol, gain2);
      const vector<Real> gradAcer = sum2Grads(gradAcer_1, gradAcer_2);
    #elif defined(ACER_NOCLIP)
      const Real gain1 = A_OPC * rho_cur;
      const vector<Real> gradAcer = pol_cur.policy_grad(act, gain1);
    #elif defined(ACER_CLIP_1)
      const Real gain1 = rho_cur>1 && A_OPC>0 ? 1*A_OPC : A_OPC*rho_cur;
      const vector<Real> gradAcer = pol_cur.policy_grad(act, gain1);
    #else
      const Real gain1 = rho_cur>5 && A_OPC>0 ? 5*A_OPC : A_OPC*rho_cur;
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

    const Real Ver = Qer*std::min((Real)1, rho_cur);
    vector<Real> gradient(nOutputs,0);
    gradient[net_indices[0]]= Qer * Qprecision;

    #if defined(ACER_CONSTRAINED)
      const vector<Real> gradDivKL = pol_cur.div_kl_grad(&pol_hat);
      const vector<Real> totalPolGrad = trust_region_update(policy_grad, gradDivKL, DKL_hardmax);
    #elif defined(ACER_ADAPTIVE)
      gradient[PenalID] = -4*std::pow(DivKL - DKL_target,3)*opt->eta;
      const vector<Real> totalPolGrad = policy_grad;
      if (thrID==1) opt->eta = out_cur[PenalID];
    #else
      const Real penalDKL = out_cur[PenalID]
      //increase if DivKL is greater than Target
      //computed as \nabla_{penalDKL} (DivKL - DKL_target)^2
      //with rough approximation that DivKL/penalDKL = penalDKL
      //(distance increases if penalty term increases, similar to PPO )
      gradient[PenalID] = 4*std::pow(DivKL - DKL_target,3)*penalDKL;
      //trust region updating
      const vector<Real>penal_grad= pol_cur.div_kl_opp_grad(&pol_hat,-penalDKL);
      const vector<Real> totalPolGrad = sum2Grads(penal_grad, policy_grad);
    #endif


    //decrease precision if error is large
    //computed as \nabla_{Qprecision} Dkl (Q^RET_dist || Q_dist)
    gradient[QPrecID] = -.5 * (Qer * Qer - 1/Qprecision);
    //adv_cur.grad(act, Qer, gradient, aInfo.bounded);
    adv_cur.grad(act, Qer * Qprecision, gradient);
    pol_cur.finalize_grad(totalPolGrad, gradient);
    //prepare Q with off policy corrections for next step:
    Q_RET = std::min((Real)1, rho_cur)*(Q_RET -A_cur -V_cur) +V_cur;
    Q_OPC = std::min(CmaxRet, rho_cur)*(Q_RET -A_cur -V_cur) +V_cur;
    //bookkeeping:
    dumpStats(Vstats[thrID], A_cur+V_cur, Qer ); //Ver
    data->Set[seq]->tuples[samp]->SquaredError = Ver*Ver;
    return gradient;
  }

  inline void offPolCorrUpdate(const Uint seq, const Uint samp, Real& Q_RET,
    Real& Q_OPC, const vector<Real> output_hat) const
  {
    const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
    //const Tuple * const t_ = data->Set[seq]->tuples[samp+1];//contains r, sNew
    const Real reward = data->standardized_reward(seq,samp+1);
    Q_RET = reward + gamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = reward + gamma*Q_OPC;
    const Real V_hat = output_hat[net_indices[0]];
    const Policy_t pol_hat = prepare_policy(output_hat);
    //Used as target: target policy, target value
    const Advantage_t adv_hat = prepare_advantage(output_hat, &pol_hat);
    //off policy stored action:
    const Action_t act = pol_hat.map_action(_t->a);//unbounded action space
    const Real actProbOnTarget = pol_hat.evalLogProbability(act);
    const Real actProbBehavior = Policy_t::evalBehavior(act,_t->mu);
    const Real C = safeExp(actProbOnTarget-actProbBehavior);
    const Real A_hat = adv_hat.computeAdvantage(act);
    //prepare rolled Q with off policy corrections for next step:
    Q_RET = std::min((Real)1, C)*(Q_RET -A_hat -V_hat) +V_hat;
    Q_OPC = std::min(CmaxRet, C)*(Q_RET -A_hat -V_hat) +V_hat;
  }

  inline Real computeImportanceWeight(const vector<Real>& out, const Tuple* const _t) const
  {
    const Policy_t pol_hat = prepare_policy(out);
    const Action_t act = pol_hat.map_action(_t->a); //to unbounded space
    const Real probTrgt = pol_hat.evalLogProbability(act);
    const Real probBeta = Policy_t::evalBehavior(act, _t->mu);
    return ACER_LAMBDA * gamma * std::min(CmaxRet, safeExp(probTrgt-probBeta));
  }

 public:
  RACER(MPI_Comm comm, Environment*const _env, Settings& sett,
    vector<Uint> net_outs, vector<Uint> pol_inds, vector<Uint> adv_inds) :
    Learner_utils(comm, _env, sett, sett.nnOutputs), CmaxRet(sett.impWeight),
    DKL_target(sett.klDivConstraint), net_outputs(net_outs),
    net_indices(count_indices(net_outs)), pol_start(pol_inds),
    adv_start(adv_inds), generators(sett.generators)
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

    printf("%lu, rmse:%.2Lg, avg_Q:%.2Lg, std_Q:%.2Lg, min_Q:%.2Lg, max_Q:%.2Lg, "
      "weight:[%.0Lg %.0Lg %.2Lg], Qprec:%.3f, penalDKL:%f, rewPrec:%f\n",
      opt->nepoch, stats.MSE, stats.avgQ, stats.stdQ,
      stats.minQ, stats.maxQ, sumWeights, sumWeightsSq, distTarget,
      Qprecision, penalDKL, data->invstd_reward);
      fflush(0);

    ofstream fs;
    fs.open("stats.txt", ios::app);
    fs<<opt->nepoch<<"\t"<<stats.MSE<<"\t"<<stats.avgQ<<"\t"<<stats.stdQ<<"\t"<<
    stats.minQ<<"\t"<<stats.maxQ<<"\t"<<sumWeights<<"\t"<<sumWeightsSq<<"\t"<<
    distTarget<<"\t"<<Qprecision<<"\t"<<penalDKL<<"\t"<<data->invstd_reward<<endl;
    fs.close();
    fs.flush();
    if (stats.epochCount % 100==0) save("policy");
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
