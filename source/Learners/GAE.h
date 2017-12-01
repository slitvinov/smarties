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
  const Real lambda, DKL_target, clip_fac = 0.2;
  const vector<Uint> pol_outputs, pol_indices;
  const Uint PenalID = pol_indices.back(), ValID = 0;

  inline Policy_t prepare_policy(const vector<Real>& out) const
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
    if(seq->state_vals.size() < 2) return;
    assert(seq->tuples.size() == seq->state_vals.size());
    assert(seq->tuples.size() == seq->action_adv.size()+1);
    const Uint N = seq->tuples.size();
    const Real vSold = seq->state_vals[N-2], vSnew = seq->state_vals[N-1];
    // delta_t = r_t+1 + gamma V(s_t+1) - V(s_t)  (pedix on r means r_t+1
    // received with transition to s_t+1, sometimes referred to as r_t)
    const Real delta = seq->tuples[N-1]->r +gamma*vSnew -vSold;
    seq->action_adv.push_back(delta);

    Real fac_lambda = lambda*gamma, fac_gamma = gamma;
    // reward of i=0 is 0, because before any action
    // adv(0) is also 0, V(0) = V(s_0)
    for (Uint i=N-2; i>0; i--) { //update all rewards before current step
      //will contain MC sum of returns:
      seq->tuples[i]->r += fac_gamma * seq->tuples[N-1]->r;
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
    if(thrID==1)  profiler->stop_start("TRAIN");
    const Sequence* const traj = data->Set[seq];
    const Real adv_est = traj->action_adv[samp+1];
    const Real val_tgt = traj->tuples[samp+1]->r;
    const vector<Real>& beta = traj->tuples[samp]->mu;

    F[0]->prepare_one(traj, samp, thrID);
    F[1]->prepare_one(traj, samp, thrID);
    const vector<Real> pol_cur = F[0]->forward<CUR>(traj, samp, thrID);
    const vector<Real> val_cur = F[1]->forward<CUR>(traj, samp, thrID);

    const Real Vst = val_cur[ValID], penalDKL = pol_cur[PenalID];
    const Policy_t pol = prepare_policy(pol_cur);
    const Action_t act = pol.map_action(traj->tuples[samp]->a);
    const Real actProbOnPolicy = pol.evalLogProbability(act);
    const Real actProbBehavior = Policy_t::evalBehavior(act, beta);
    const Real rho_cur= min(MAX_IMPW, safeExp(actProbOnPolicy-actProbBehavior));
    const Real DivKL=pol.kl_divergence_opp(beta);
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

    vector<Real> grad(F[0]->nOutputs(), 0);
    pol.finalize_grad(totalPolGrad, grad);
    grad[PenalID] = 4*std::pow(DivKL - DKL_target,3)*penalDKL;

    //bookkeeping:
    Vstats[thrID].dumpStats(Vst, val_tgt - Vst);
    F[0]->backward(grad, samp, thrID);
    F[1]->backward({val_tgt - Vst}, samp, thrID);
    F[0]->gradient(thrID);
    F[1]->gradient(thrID);

    if(thrID==1)  profiler->stop_start("SLP");
  }

public:
  GAE(Environment*const _env, Settings& sett, vector<Uint> pol_outs) :
    Learner_onPolicy(_env, sett), lambda(sett.lambda),
    DKL_target(sett.klDivConstraint), pol_outputs(pol_outs),
    pol_indices(count_indices(pol_outs)) { }

  //called by scheduler:
  void select(const Agent& agent) override
  {
    const int thrID= omp_get_thread_num();
    if(thrID==1) profiler->stop_start("SELECT");
    Sequence*const curr_seq = data->inProgress[agent.ID];
    data->add_state(agent);

    if(agent.Status != 2) { //non terminal state
      //Compute policy and value on most recent element of the sequence:
      const vector<Real> pol=F[0]->forward_agent<CUR>(curr_seq, agent, thrID);
      const vector<Real> val=F[1]->forward_agent<CUR>(curr_seq, agent, thrID);

      curr_seq->state_vals.push_back(val[0]);
      const Policy_t policy = prepare_policy(pol);
      const vector<Real> beta = policy.getBeta();
      agent.a->set(policy.finalize(bTrain, &generators[thrID], beta));
      data->add_action(agent, beta);
    } else
      curr_seq->state_vals.push_back(0); // Assign value of term state to 0

    updateGAE(curr_seq);

    //advance counters of available data for training
    if(agent.Status==2) {
      data->terminate_seq(agent);
      lock_guard<mutex> lock(buffer_mutex);
      cntHorizon += curr_seq->ndata();
      cntTrajectories++;
    }
    if(thrID==0) profiler_ext->stop_start("COMM");
    if(thrID==1) profiler->pop_stop();
  }

  void getMetrics(ostringstream&fileOut, ostringstream&screenOut) const
  {
    Uint params_inds = F[0]->net->layers.back()->n1stBias;
    const Real penalDKL = std::exp(F[0]->net->biases[params_inds]);
    screenOut<<" penalDKL:"<<penalDKL;
    fileOut<<" "<<penalDKL;
  }
};

class GAE_cont : public GAE<Gaussian_policy, vector<Real> >
{
  static vector<Uint> count_pol_outputs(const ActionInfo*const aI)
  {
    return vector<Uint>{aI->dim, aI->dim, 1};
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

  GAE_cont(Environment*const _env, Settings & settings) :
  GAE(_env, settings, count_pol_outputs(&_env->aI))
  {
    printf("Continuous-action GAE\n");
    F.push_back(new Approximator("policy", settings, input, data));
    F.push_back(new Approximator("value", settings, input, data));
    #ifndef simpleSigma
      Builder build_pol = F[0]->buildFromSettings(settings, 2*aInfo.dim);
    #else
      Builder build_pol = F[0]->buildFromSettings(settings,   aInfo.dim);
    #endif
    Builder build_val = F[1]->buildFromSettings(settings, 1 );

    #ifdef simpleSigma //add stddev layer
      build_pol.addParamLayer(aInfo.dim, "Linear", -2*std::log(greedyEps));
    #endif
    build_pol.addParamLayer(1, "Exp", 0); //add klDiv penalty coefficient layer

    F[0]->build_network(build_pol);
    F[1]->build_network(build_val);

    //set initial value for klDiv penalty coefficient (was last added layer)
    const Uint penalparid = F[0]->net->layers.back()->n1stBias;
    F[0]->net->biases[penalparid] = 0;

    F[0]->build_finalize(build_pol);
    F[1]->build_finalize(build_val);
  }
};

class GAE_disc : public GAE<Discrete_policy, Uint>
{
  static vector<Uint> count_pol_outputs(const ActionInfo*const aI)
  {
    return vector<Uint>{aI->maxLabel, 1};
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
  GAE_disc(Environment*const _env, Settings& settings) :
  GAE(_env, settings, count_pol_outputs(&_env->aI))
  {
    printf("Discrete-action GAE\n");
    F.push_back(new Approximator("policy", settings, input, data));
    F.push_back(new Approximator("value", settings, input, data));
    Builder build_pol = F[0]->buildFromSettings(settings, aInfo.maxLabel);
    Builder build_val = F[1]->buildFromSettings(settings, 1 );

    build_pol.addParamLayer(1, "Exp", 0); //add klDiv penalty coefficient layer

    F[0]->build_network(build_pol);
    F[1]->build_network(build_val);

    //set initial value for klDiv penalty coefficient (was last added layer)
    const Uint penalparid = F[0]->net->layers.back()->n1stBias;
    F[0]->net->biases[penalparid] = 0;

    F[0]->build_finalize(build_pol);
    F[1]->build_finalize(build_val);
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
