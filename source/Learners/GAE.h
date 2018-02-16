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
#include "../Network/Builder.h"
#define PPO_PENALKL
#define PPO_CLIPPED
//#define IGNORE_CRITIC
#define simpleSigma

template<typename Policy_t, typename Action_t>
class GAE : public Learner_onPolicy
{
protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  const Real lambda, learnR, DKL_target, clip_fac = 0.2;
  const vector<Uint> pol_outputs, pol_indices;
  mutable vector<long double> cntPenal, valPenal;

  inline Policy_t prepare_policy(const Rvec& out) const
  {
    return Policy_t(pol_indices, &aInfo, out);
  }

  void Train_BPTT(const Uint seq, const Uint thrID) const override
  {
    die("not allowed");
  }

  void updateGAE(Sequence*const seq) const
  {
    //this is only triggered by t = 0 (or truncated trajectories)
    // at t=0 we do not have a reward, and we cannot compute delta
    //(if policy was updated after prev action we treat next state as initial)
    if(seq->state_vals.size() < 2) {
      return;
    }
    assert(seq->tuples.size() == seq->state_vals.size());
    assert(seq->tuples.size() == 2+seq->Q_RET.size());
    assert(seq->tuples.size() == 2+seq->action_adv.size());
    const Uint N = seq->tuples.size();
    const Real vSold = seq->state_vals[N-2], vSnew = seq->state_vals[N-1];
    // delta_t = r_t+1 + gamma V(s_t+1) - V(s_t)  (pedix on r means r_t+1
    // received with transition to s_t+1, sometimes referred to as r_t)
    const Real delta = seq->tuples[N-1]->r +gamma*vSnew -vSold;
    seq->action_adv.push_back(0);
    seq->Q_RET.push_back(0);

    Real fac_lambda = 1, fac_gamma = 1;
    // reward of i=0 is 0, because before any action
    // adv(0) is also 0, V(0) = V(s_0)
    for (int i=N-2; i>=0; i--) { //update all rewards before current step
      //will contain MC sum of returns:
      seq->Q_RET[i] += fac_gamma * seq->tuples[N-1]->r;
      #ifndef IGNORE_CRITIC
        seq->action_adv[i] += fac_lambda * delta;
      #else
        seq->action_adv[i] += fac_gamma * seq->tuples[N-1]->r;
      #endif
      fac_lambda *= lambda*gamma;
      fac_gamma *= gamma;
    }
  }

  void Train(const Uint seq, const Uint samp,const Uint thrID) const override
  {
    if(thrID==1)  profiler->stop_start("FWD");
    const Sequence* const traj = data->Set[seq];
    const Real adv_est = traj->action_adv[samp];
    const Real val_tgt = traj->Q_RET[samp];
    const Rvec& beta = traj->tuples[samp]->mu;

    F[0]->prepare_one(traj, samp, thrID);
    F[1]->prepare_one(traj, samp, thrID);
    const Rvec pol_cur = F[0]->forward(traj, samp, thrID);
    const Rvec val_cur = F[1]->forward(traj, samp, thrID);

    if(thrID==1)  profiler->stop_start("CMP");

    const Real Vst = val_cur[0], penalDKL = valPenal[0];
    const Policy_t pol = prepare_policy(pol_cur);
    const Action_t act = pol.map_action(traj->tuples[samp]->a);
    const Real actProbOnPolicy = pol.evalLogProbability(act);
    const Real actProbBehavior = Policy_t::evalBehavior(act, beta);
    const Real rho_cur = safeExp(actProbOnPolicy-actProbBehavior);
    const Real DivKL=pol.kl_divergence_opp(beta);
    //if ( thrID==1 ) printf("%u %u : %f DivKL:%f %f %f\n", nOutputs, PenalID, penalDKL, DivKL, completed[workid]->policy[samp][1], output[2]);

    Real gain = rho_cur*adv_est;
    #ifdef PPO_CLIPPED
      bool gainZero = false;
      if (adv_est > 0 && rho_cur > 1+clip_fac) gainZero = true; //gain = 0;
      if (adv_est < 0 && rho_cur < 1-clip_fac) gainZero = true; //gain = 0;
      // if off policy, skip zero-gradient backprop
      // pro: avoid messing adam statistics
      if(gainZero) return resample(thrID);
    #endif

    cntPenal[thrID+1]++;
    //grad[PenalID] = 4*std::pow(DivKL - DKL_target,3)*penalDKL;
    if(DivKL > 1.5 * DKL_target) valPenal[thrID+1] += penalDKL; //double
    if(DivKL < DKL_target / 1.5) valPenal[thrID+1] -= penalDKL/2; //half


    #ifdef PPO_PENALKL
      const Rvec policy_grad = pol.policy_grad(act, gain);
      const Rvec penal_grad = pol.div_kl_opp_grad(beta, -penalDKL);
      Rvec totalPolGrad = sum2Grads(penal_grad, policy_grad);
    #else //we still learn the penal coef, for simplicity, but no effect
      Rvec totalPolGrad = pol.policy_grad(act, gain);
    #endif

    Rvec grad(F[0]->nOutputs(), 0);
    pol.finalize_grad(totalPolGrad, grad);

    //bookkeeping:
    Vstats[thrID].dumpStats(Vst, val_tgt - Vst);
    if(thrID==1)  profiler->stop_start("BCK");

    F[0]->backward(grad, samp, thrID);
    F[0]->gradient(thrID);
    F[1]->backward({val_tgt - Vst}, samp, thrID);
    F[1]->gradient(thrID);
  }

public:
  GAE(Environment*const _env, Settings& _set, vector<Uint> pol_outs) :
    Learner_onPolicy(_env, _set), lambda(_set.lambda), learnR(_set.learnrate),
    #ifdef PPO_CLIPPED
    DKL_target(_set.klDivConstraint *std::sqrt(nA)), //small penalty, still better perf
    #else
    DKL_target(_set.klDivConstraint),
    #endif
    pol_outputs(pol_outs), pol_indices(count_indices(pol_outs)),
    cntPenal(nThreads+1, 0), valPenal(nThreads+1, 0) {
    valPenal[0] = 1./DKL_target;
    //valPenal[0] = 1.;
  }

  //called by scheduler:
  void select(const Agent& agent) override
  {
    const int thrID= omp_get_thread_num();
    Sequence*const curr_seq = data->inProgress[agent.ID];
    data->add_state(agent);

    if(agent.Status != 2) { //non terminal state
      //Compute policy and value on most recent element of the sequence:
      const Rvec pol=F[0]->forward_agent(curr_seq, agent, thrID);
      const Rvec val=F[1]->forward_agent(curr_seq, agent, thrID);

      curr_seq->state_vals.push_back(val[0]);
      Policy_t policy = prepare_policy(pol);
      const Rvec beta = policy.getBeta();
      agent.a->set(policy.finalize(bTrain, &generators[thrID], beta));
      data->add_action(agent, beta);
    } else
      curr_seq->state_vals.push_back(0); // Assign value of term state to 0

    updateGAE(curr_seq);

    //advance counters of available data for training
    if(agent.Status==2) data->terminate_seq(agent);
  }

  void getMetrics(ostringstream& buff) const
  {
    buff<<" "<<std::setw(6)<<std::setprecision(4)<<valPenal[0];
  }
  void getHeaders(ostringstream& buff) const
  {
    buff <<"| beta ";
  }

  void prepareGradient()
  {
    const bool bWasPrepareReady = updateComplete;

    Learner_onPolicy::prepareGradient();

    if(not bWasPrepareReady) return;

    cntPenal[0] = 0;
    for(Uint i=1; i<=nThreads; i++) {
      cntPenal[0] += cntPenal[i]; cntPenal[i] = 0;
    }
    const Real fac = learnR/cntPenal[0]; // learnRate*grad/N
    cntPenal[0] = 0;
    for(Uint i=1; i<=nThreads; i++) {
        valPenal[0] += fac*valPenal[i];
        valPenal[i] = 0;
    }
    if(valPenal[0] <= nnEPS) valPenal[0] = nnEPS;
  }
};

class GAE_cont : public GAE<Gaussian_policy, Rvec >
{
  static vector<Uint> count_pol_outputs(const ActionInfo*const aI)
  {
    return vector<Uint>{aI->dim, aI->dim};
  }
  static vector<Uint> count_pol_starts(const ActionInfo*const aI)
  {
    const vector<Uint> indices = count_indices(count_pol_outputs(aI));
    return vector<Uint>{indices[0], indices[1]};
  }

 public:
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim;
  }

  GAE_cont(Environment*const _env, Settings & _set) :
  GAE(_env, _set, count_pol_outputs(&_env->aI))
  {
    printf("Continuous-action GAE\n");
    F.push_back(new Approximator("policy", _set, input, data));
    F.push_back(new Approximator("value", _set, input, data));
    #ifndef simpleSigma
      Builder build_pol = F[0]->buildFromSettings(_set, {2*aInfo.dim});
    #else
      Builder build_pol = F[0]->buildFromSettings(_set,   {aInfo.dim});
    #endif
    Builder build_val = F[1]->buildFromSettings(_set, {1} );

    #ifdef simpleSigma //add stddev layer
      build_pol.addParamLayer(aInfo.dim, "Linear", -2*std::log(greedyEps));
    #endif
    //add klDiv penalty coefficient layer initialized to 1
    //build_pol.addParamLayer(1, "Exp", 1);

    F[0]->initializeNetwork(build_pol);
    //_set.learnrate *= 2;
    F[1]->initializeNetwork(build_val);
  }
};

class GAE_disc : public GAE<Discrete_policy, Uint>
{
  static vector<Uint> count_pol_outputs(const ActionInfo*const aI)
  {
    return vector<Uint>{aI->maxLabel};
  }
  static vector<Uint> count_pol_starts(const ActionInfo*const aI)
  {
    const vector<Uint> indices = count_indices(count_pol_outputs(aI));
    return vector<Uint>{indices[0]};
  }
 public:
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return aI->maxLabel;
  }

 public:
  GAE_disc(Environment*const _env, Settings& _set) :
  GAE(_env, _set, count_pol_outputs(&_env->aI))
  {
    printf("Discrete-action GAE\n");
    F.push_back(new Approximator("policy", _set, input, data));
    F.push_back(new Approximator("value", _set, input, data));
    Builder build_pol = F[0]->buildFromSettings(_set, aInfo.maxLabel);
    Builder build_val = F[1]->buildFromSettings(_set, 1 );

    //build_pol.addParamLayer(1,"Exp",1); //add klDiv penalty coefficient layer

    F[0]->initializeNetwork(build_pol);
    F[1]->initializeNetwork(build_val);
  }
};

#if 0

// Update network from sampled observation `obs', part of sequence `seq'
void Train(const Uint seq, const Uint obs, const Uint thrID) const override
{
  const Sequence*const traj = data->Set[seq];          // fetch sampled sequence
  const Real advantage_obs  = traj->action_adv[obs+1];// observed advantage
  const Real value_obs      = traj->tuples[obs+1]->r; // observed state val
  const Rvec mu     = traj->tuples[obs]->mu;  // policy used for sample
  const Rvec action = traj->tuples[obs]->a;   // sample performed act

  // compute current policy and state-value-estimate for sampled state
  const Rvec out_policy = policyNet->forward(traj, samp, thrID);
  const Rvec out_value  =  valueNet->forward(traj, samp, thrID);

  //compute gradient of state-value est. and backpropagate value net
  const Real Vst_est  = out_value[0];           // estimated state value
  const Rvec  value_grad = {value_obs - Vst_est};
   valueNet->backward(value_grad, samp, thrID);

  //Create action & policy objects: generalize discrete, gaussian, lognorm pols
  const Policy_t pol = prepare_policy(pol_cur);//current state policy
  const Action_t act = pol.map_action(action); //map to pol space (eg. discrete)

  // compute importance sample rho = pol( a_t | s_t ) / mu( a_t | s_t )
  const Real actProbOnPolicy =       pol.evalLogProbability(act);
  const Real actProbBehavior = Policy_t::evalLogProbability(act, mu);
  const Real rho = std::exp(actProbOnPolicy-actProbBehavior);

  //compute policy gradient and backpropagate pol net
  const Real gain = rho * advantage_obs;
  const Rvec  policy_grad = pol.policy_grad(act, gain);
  policyNet->backward(policy_grad, samp, thrID);
}

#endif
//#ifdef INTEGRATEANDFIREMODEL
//  inline Lognormal_policy prepare_policy(const Rvec& out) const
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
