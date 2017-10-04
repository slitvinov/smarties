/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */
#pragma once
#include "Learner_onPolicy.h"
#include "../Math/Lognormal_policy.h"
#include "../Math/Gaussian_policy.h"
#define PPO_PENALKL
#define PPO_CLIPPED
//#define IGNORE_CRITIC
#define simpleSigma

template<typename Policy_t, typename Action_t>
class GAE : public Learner_onPolicy
{
protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Real lambda = 0.95, DKL_target = 0.01, clip_fac = 0.2;
  //#ifdef INTEGRATEANDFIRESHARED
  //  const vector<Uint> net_outputs = {nA, 1};
  //#else
  vector<Uint> net_outputs, net_indices;
  const vector<Uint> pol_start;
  const Uint PenalID = net_indices.back(), ValID = net_indices[1];

  inline Policy_t prepare_policy(const vector<Real>& out) const
  {
    return Policy_t(pol_start, &aInfo, out);
  }

  void Train_BPTT(const Uint seq, const Uint thrID) const override
  {
    die("not allowed");
  }

  void updateGAE(const int workid, const Real reward, const Real delta) const
  {
    Real fac_lambda = lambda*gamma, fac_gamma = gamma;
    work[workid]->rewards.push_back(reward);
    work[workid]->GAE.push_back(delta);
    for (Uint i=2; i<=work[workid]->GAE.size(); i++) {
      const Uint ind = work[workid]->GAE.size() - i;
      work[workid]->rewards[ind] += fac_gamma*reward;
      #ifndef IGNORE_CRITIC
      work[workid]->GAE[ind] += fac_lambda*delta;
      #else
      work[workid]->GAE[ind] += fac_gamma*reward;
      #endif
      fac_lambda *= lambda*gamma;
      fac_gamma *= gamma;
    }
  }

  void Train(const Uint workid,const Uint samp,const Uint thrID) const override
  {
    if(thrID==1)  profiler->stop_start("TRAIN");
    vector<Real> output(nOutputs), grad(nOutputs,0);
    const Real adv_est = completed[workid]->GAE[samp];
    const Real val_tgt = completed[workid]->rewards[samp];
    const vector<Real>& beta = completed[workid]->policy[samp];
    const Uint nMaxBPTT = MAX_UNROLL_BFORE;
    //printf("%u %u %u %f %f \n", workid, samp, thrID, val_tgt, adv_est);
    const Uint nRecurr = bRecurrent ? min(nMaxBPTT,samp)+1        : 1;
    const Uint iRecurr = bRecurrent ? max(nMaxBPTT,samp)-nMaxBPTT : samp;

    net->prepForBackProp(series_1[thrID], nRecurr);
    vector<Activation*>& series = *(series_1[thrID]);
    assert(series.size()==nRecurr);
    for (Uint k=iRecurr, j=0; k<samp+1; k++, j++) {
      net->seqPredict_inputs(completed[workid]->observations[k], series[j]);
      if(k==samp) { //all are loaded: execute the whole loop:
        assert(j==nRecurr-1);
        net->seqPredict_execute(series, series);
        //extract the only output we actually correct:
        net->seqPredict_output(output, series[j]); //the humanity!
      }
    }

    const Real Vst = output[ValID];
    const Policy_t pol = prepare_policy(output);
    const Action_t act = pol.map_action(completed[workid]->actions[samp]);
    const Real actProbOnPolicy = pol.evalLogProbability(act);
    const Real actProbBehavior = Policy_t::evalBehavior(act, beta);
    const Real rho_cur= min(MAX_IMPW, safeExp(actProbOnPolicy-actProbBehavior));
    const Real DivKL=pol.kl_divergence_opp(beta), penalDKL=output[PenalID];
    //if ( thrID==1 ) printf("%u %u : %f DivKL:%f %f %f\n", nOutputs, PenalID, penalDKL, DivKL, completed[workid]->policy[samp][1], output[2]);

    Real gain = rho_cur*adv_est;
    #ifdef PPO_CLIPPED
      if (adv_est > 0 && rho_cur > 1+clip_fac) gain = 0;
      if (adv_est < 0 && rho_cur < 1-clip_fac) gain = 0;
    #endif

    #ifdef PPO_PENALKL
      const vector<Real> policy_grad = pol.policy_grad(act, gain);
      const vector<Real> penal_grad = pol.div_kl_opp_grad(beta, -penalDKL);
      vector<Real> totalPolGrad = sum2Grads(penal_grad, policy_grad);
    #else //we still learn the penal coef, for simplicity, but no effect
      if(gain==0) {
        if(thrID==1)  profiler->stop_start("SLP");
        return; //if 0 pol grad dont backprop
      }
      vector<Real> totalPolGrad = pol.policy_grad(act, gain);
    #endif

    grad[ValID] = val_tgt - Vst;
    pol.finalize_grad(totalPolGrad, grad);
    grad[PenalID] = 4*std::pow(DivKL - DKL_target,3)*penalDKL;

    //bookkeeping:
    dumpStats(Vstats[thrID], Vst, val_tgt - Vst);
    statsGrad(avgGrad[thrID+1], stdGrad[thrID+1], cntGrad[thrID+1], grad);
    clip_grad(grad, stdGrad[0]);
    net->setOutputDeltas(grad, series.back());

    if(thrID==1)  profiler->stop_start("BCK");
    if (thrID==0) net->backProp(series, net->grad);
    else net->backProp(series, net->Vgrad[thrID]);
    if(thrID==1)  profiler->stop_start("SLP");
  }

public:
  GAE(MPI_Comm comm, Environment*const _env, Settings& settings,
  vector<Uint> net_outs, vector<Uint> pol_inds) :
  Learner_onPolicy(comm, _env, settings, settings.nnOutputs),
  net_outputs(net_outs), net_indices(count_indices(net_outs)),
  pol_start(pol_inds) { }

  //called by scheduler:
  void select(const int agentId, const Agent& agent) override
  {
    const int thrID= omp_get_thread_num(), workid= retrieveAssignment(agentId);
    //printf("Thread %d working with agent %d on task %d with status %d\n", thrid, agentId, workid, agent.Status); fflush(0);
    if(workid<0) die("workspace not allocated.");
    //printf("(%lu %lu %lu)\n", work[workid]->series.size(), work[workid]->actions.size(), work[workid]->rewards.size()); fflush(0);
    if(agent.Status==2) { //terminal state
      data->writeData(agentId, agent, vector<Real>(policyVecDim,0));
      if(work[workid]->Vst.size() == 0) { //empty trajectory
        work[workid]->clear(); //reset workspace for other trajectory
        return;
      }
      // V_s_term := 0, therefore delta_term = r_t+1 - V(s_t)
      const Real delta = agent.r - work[workid]->Vst.back();
      assert(work[workid]->GAE.size() == work[workid]->rewards.size());
      assert(work[workid]->GAE.size()+1 == work[workid]->Vst.size());
      updateGAE(workid, agent.r, delta);
      work[workid]->done = 1;
      addTasks(work[workid]);
      return;
    }
    if(thrID==1) profiler->stop_start("FWD");
    vector<Real> output(nOutputs);
    const vector<Real> input = data->standardize(agent.s->copy_observed());
    //if required, chain together nAppended obs to compose state
    assert(!nAppended); //not supported

    if(agent.Status==1) net->predict(input, output, currAct[thrID]);
    else { // if i'm using RNN i need to load recur connections (else no effect)
      prevAct[thrID]->loadMemory(net->mem[agentId]); // prevAct[thrID],
      net->predict(input, output, prevAct[thrID],currAct[thrID]);
    }
    currAct[thrID]->storeMemory(net->mem[agentId]);

    if(thrID==0) profiler_ext->stop_start("WORK");
    const Real val = output[ValID];
    const Policy_t pol = prepare_policy(output);
    const vector<Real> beta = pol.getBeta();
    const Action_t act = pol.finalize(bTrain, gen, beta);
    agent.a->set(act);

    //treated as first in two circumstances:
    // - if actually initial state
    // - or if policy was updated after prev action
    // (in PPO T_horizon is not linked to Term. states)
    if( agent.Status != 1 && work[workid]->Vst.size() != 0 )  {
      // delta_t = r_t+1 + gamma V(s_t+1) - V(s_t)  (pedix on r means r_t+1
      // received with transition to s_t+1, sometimes referred to as r_t)
      const Real delta = agent.r +gamma*val -work[workid]->Vst.back();
      updateGAE(workid, agent.r, delta);
    }

    assert(work[workid]->GAE.size() == work[workid]->Vst.size()); //b4 new Vst
    work[workid]->push_back(input, agent.a->vals, beta, val);
    data->writeData(agentId, agent, beta);
    if(thrID==0) profiler_ext->stop_start("COMM");
    if(thrID==1) profiler->pop_stop();
  }

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
      long double ary[4] = {stats.MSE, stats.dCnt, stats.avgQ, stats.stdQ};
      MPI_Allreduce(MPI_IN_PLACE,ary,4,MPI_LONG_DOUBLE,MPI_SUM,mastersComm);
      stats.MSE=ary[0]; stats.dCnt=ary[1]; stats.avgQ=ary[2]; stats.stdQ=ary[3];
      MPI_Allreduce(MPI_IN_PLACE, &stats.minQ, 1, MPI_LONG_DOUBLE, MPI_MIN, mastersComm);
      MPI_Allreduce(MPI_IN_PLACE, &stats.maxQ, 1, MPI_LONG_DOUBLE, MPI_MAX, mastersComm);
    }

    stats.epochCount++;
    epochCounter = stats.epochCount;
    const long double sum = stats.avgQ, sumsq = stats.stdQ, cnt = stats.dCnt;
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
    const Real penalDKL = exp(net->biases[net->layers.back()->n1stBias]);

    printf("%lu, rmse:%.2Lg, avgQ:%.2Lg, stdQ:%.2Lg, minQ:%.2Lg, maxQ:%.2Lg, "\
      "weight:[%.0Lg %.0Lg %.2Lg], penalDKL:%.3f\n",
      opt->nepoch, stats.MSE, stats.avgQ, stats.stdQ, stats.minQ, stats.maxQ,
      sumWeights, sumWeightsSq, distTarget, penalDKL);
      fflush(0);
    ofstream fs;
    fs.open("stats.txt", ios::app);
    fs<<opt->nepoch<<"\t"<<stats.MSE<<"\t"<<stats.avgQ<<"\t"<<stats.stdQ<<"\t"<<
      stats.minQ<<"\t"<<stats.maxQ<<"\t"<<sumWeights<<"\t"<<sumWeightsSq<<"\t"<<
      distTarget<<"\t"<<penalDKL<<endl;
    fs.close(); fs.flush();
    if (stats.epochCount % 100==0) save("policy");
  }
};

class GAE_cont : public GAE<Gaussian_policy, vector<Real> >
{
  static vector<Uint> count_outputs(const ActionInfo& aI)
  {
    return vector<Uint>{aI.dim, 1, aI.dim, 1};
  }
  static vector<Uint> count_pol_starts(const ActionInfo& aI)
  {
    const vector<Uint> indices = count_indices(count_outputs(aI));
    return vector<Uint>{indices[0], indices[2]};
  }

 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    return 1 + aI->dim + aI->dim + 1;
  }

  GAE_cont(MPI_Comm comm, Environment*const _env, Settings & settings) :
  GAE(comm, _env, settings, count_outputs(_env->aI),
  count_pol_starts(_env->aI))
  {
    settings.splitLayers = 9; // all!
    Builder build(settings);

    vector<Uint> outs{nA, 1};
    #ifndef simpleSigma
      outs.push_back(nA);
    #endif
    build.stackSimple(vector<Uint>{nInputs}, outs );
    #ifdef simpleSigma //add stddev layer
      build.addParamLayer(nA, "Linear", -2*std::log(greedyEps));
    #endif
    //add klDiv penalty coefficient layer
    build.addParamLayer(1, "Exp", 0);

    net = build.build();
    //set initial value for klDiv penalty coefficient
    Uint penalparid = net->layers.back()->n1stBias; //(was last added layer)
    net->biases[penalparid] = -std::log(settings.klDivConstraint);

    finalize_network(build);
    printf("Continuous-action GAE: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    policyVecDim = 2*nA;
  }
};

class GAE_disc : public GAE<Discrete_policy, Uint>
{
  static vector<Uint> count_outputs(const ActionInfo*const aI)
  {
    return vector<Uint>{aI->maxLabel, 1, 1};
  }
  static vector<Uint> count_pol_starts(const ActionInfo*const aI)
  {
    const vector<Uint> indices = count_indices(count_outputs(aI));
    return vector<Uint>{indices[0]};
  }
 public:
  static Uint getnOutputs(const ActionInfo*const aI)
  {
    return 1 + aI->maxLabel + 1;
  }

 public:
  GAE_disc(MPI_Comm comm, Environment*const _env, Settings & settings) :
  GAE(comm, _env, settings, count_outputs(&_env->aI),
  count_pol_starts(&_env->aI))
  {
    settings.splitLayers = 9; // all!
    Builder build(settings);

    build.stackSimple(vector<Uint>{nInputs}, vector<Uint>{nA, 1} );
    build.addParamLayer(1, "Exp", 0); //add klDiv penalty coefficient layer

    net = build.build();

    //set initial value for klDiv penalty coefficient
    Uint penalparid = net->layers.back()->n1stBias; //(was last added layer)
    net->biases[penalparid] = 0;

    finalize_network(build);
    printf("Discrete-action GAE: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    policyVecDim = nA;
  }
};

//#ifdef INTEGRATEANDFIREMODEL
//  inline Lognormal_policy prepare_policy(const vector<Real>& out) const
//  {
//    return Lognormal_policy(net_indices[0], net_indices[1], nA, out);
//  }
//#else
/*
settings.splitLayers = 9; // all!
Builder build(settings);

build.stackSimple(vector<Uint>{nInputs}, vector<Uint>{nA, 1});
//add stddev layer
build.addParamLayer(nA, "Linear", -2*std::log(greedyEps));
//add klDiv penalty coefficient layer
build.addParamLayer(1, "Exp", 0);

net = build.build();

//set initial value for klDiv penalty coefficient
Uint penalparid = net->layers.back()->n1stBias; //(was last added layer)
net->biases[penalparid] = std::log(1/settings.klDivConstraint);

finalize_network(build);

printf("GAE: Built network with outputs: %s %s\n",
  print(net_indices).c_str(),print(net_outputs).c_str());
*/
