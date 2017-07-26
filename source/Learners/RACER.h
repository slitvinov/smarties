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
#define ACER_AGGRESSIVE

class RACER : public Learner_utils
{
  #ifdef ACER_TABC
  const Real truncation;
  #endif
  const Real delta;
  const Uint nA, nL;
  std::vector<std::mt19937>& generators;
  #if defined ACER_RELAX // output V(s), P(s), pol(s), prec(s)
    vector<Uint> net_outputs = {1, nL, nA, nA};
    vector<Uint> net_indices = {0,  1, 1+nL, 1+nL+nA};
  #elif defined ACER_SAFE // output V(s), P(s), pol(s), mu(s)
    vector<Uint> net_outputs = {1, nL, nA, nA};
    vector<Uint> net_indices = {0,  1, 1+nL, 1+nL+nA};
  #else // output V(s), P(s), pol(s), prec(s), mu(s)
    vector<Uint> net_outputs = {1, nL, nA, nA, nA};
    vector<Uint> net_indices = {0,  1, 1+nL, 1+nL+nA, 1+nL+2*nA};
  #endif
  #ifdef FEAT_CONTROL
  const ContinuousSignControl* task;
  #endif

  inline Gaussian_policy prepare_policy(const vector<Real>& out) const
  {
    #if defined ACER_SAFE
    return Gaussian_policy(net_indices[2], nA, out);
    #else
    return Gaussian_policy(net_indices[2], net_indices[3], nA, out);
    #endif
  }
  inline Quadratic_advantage prepare_advantage(const vector<Real>& out,
      const Gaussian_policy*const pol) const
  {
    #if defined ACER_RELAX
    return Quadratic_advantage(net_indices[1], nA, nL, out, pol);
    #else
    #if defined ACER_SAFE
    return Quadratic_advantage(net_indices[1],net_indices[3],nA,nL,out,pol);
    #else
    return Quadratic_advantage(net_indices[1],net_indices[4],nA,nL,out,pol);
    #endif
    #endif
  }

  void Train_BPTT(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

  template<int bUpdateOPC>
  inline vector<Real> compute(const Uint seq, const Uint samp, Real& Q_RET,
    Real& Q_OPC, const vector<Real>& out_cur, const vector<Real>& out_hat,
    const Real rGamma, const Uint thrID) const
  {
    const Real anneal = opt->nepoch>epsAnneal ? 1 : Real(opt->nepoch)/epsAnneal;
    const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
    const Tuple * const t_ = data->Set[seq]->tuples[samp+1]; //contains r, sNew
    Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = t_->r + rGamma*Q_OPC;
    //get everybody camera ready:
    const Real V_cur = out_cur[net_indices[0]];
    const Gaussian_policy pol_cur = prepare_policy(out_cur);
    const Gaussian_policy pol_hat = prepare_policy(out_hat);
    const Real V_hat = out_hat[net_indices[0]];

    #ifndef ACER_AGGRESSIVE
      //Used for update of value: target policy, current value
      const Quadratic_advantage adv_cur = prepare_advantage(out_cur, &pol_hat);
      //Used for update of policy: current policy, target value
      const Quadratic_advantage adv_pol = prepare_advantage(out_hat, &pol_cur);
      const Real A_OPC = Q_OPC - V_hat;
    #else
      const Quadratic_advantage adv_cur = prepare_advantage(out_cur, &pol_cur);
      const Quadratic_advantage adv_pol = prepare_advantage(out_cur, &pol_cur);
      const Real A_OPC = Q_OPC - V_cur;
    #endif

    //off policy stored action and on-policy sample:
    const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
    const Real actProbOnPolicy = pol_cur.evalLogProbability(act);
    const Real actProbBehavior = Gaussian_policy::evalBehavior(act,_t->mu);
    const Real rho_cur = safeExp(actProbOnPolicy-actProbBehavior);
    const Real varCritic = adv_pol.advantageVariance();
    const Real A_cur = adv_cur.computeAdvantage(act);
    const Real A_cov = adv_pol.computeAdvantage(act);
    const Real Qer = Q_RET -A_cur -V_cur;
    const Real iEpsA = A_cov * std::pow((A_OPC-A_cov)/(varCritic+2.2e-16), 2);
    const Real eta = anneal * iEpsA * A_OPC * safeExp( -iEpsA * A_cov);

    #ifdef ACER_PENALIZER
      const Real cotrolVar = A_cov;
    #else
      const Real cotrolVar = 0;
    #endif

    //compute quantities needed for trunc import sampl with bias correction
    #ifdef ACER_TABC
      const vector<Real> pol = pol_cur.sample(&generators[thrID]);
      const Real polProbOnPolicy = pol_cur.evalLogProbability(pol);
      const Real polProbBehavior = Gaussian_policy::evalBehavior(pol,_t->mu);
      const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
      const Real A_pol = adv_pol.computeAdvantage(pol);
      const Real gain1 = A_OPC*min(rho_cur,truncation) -eta*rho_cur*cotrolVar;
      const Real gain2 = A_pol*max(0.,1.-truncation/rho_pol);

      const vector<Real> gradAcer_1 = pol_cur.policy_grad(act, gain1);
      const vector<Real> gradAcer_2 = pol_cur.policy_grad(pol, gain2);
      const vector<Real> gradAcer = sum2Grads(gradAcer_1, gradAcer_2);
    #else
      const Real gain1 = A_OPC * rho_cur - eta * rho_cur * cotrolVar;
      const vector<Real> gradAcer = pol_cur.policy_grad(act, gain1);
    #endif

    #ifdef ACER_PENALIZER
      const vector<Real> gradC = pol_cur.control_grad(&adv_pol, eta);
      const vector<Real> policy_grad = sum2Grads(gradAcer, gradC);
    #else
      const vector<Real> policy_grad = gradAcer;
    #endif

    //trust region updating
    const vector<Real> gradDivKL = pol_cur.div_kl_grad(&pol_hat);
    const vector<Real> trust_grad=
      trust_region_update(policy_grad,gradDivKL,delta);

    const Real Ver = Qer*std::min(1.,rho_cur);
    vector<Real> gradient(nOutputs,0);
    gradient[net_indices[0]]= Qer + Ver;
    adv_cur.grad(act, Qer, gradient);
    pol_cur.finalize_grad(trust_grad, gradient);

    if(bUpdateOPC) //prepare Q with off policy corrections for next step:
    {
      const Real C = std::min((Real)1.,std::pow(rho_cur,1./nA));
      #ifdef ACER_AGGRESSIVE
        Q_RET = C*ACER_LAMBDA*(Q_RET -A_cur -V_cur) +V_hat;
        Q_OPC = C*ACER_LAMBDA*(Q_RET -A_cur -V_cur) +V_cur;
      #else
        //Used as target: target policy, target value
        const Quadratic_advantage adv_hat = prepare_advantage(out_hat,&pol_hat);
        const Real A_hat = adv_hat.computeAdvantage(act);
        Q_RET = C*ACER_LAMBDA*(Q_RET -A_hat -V_hat) +V_hat;
        Q_OPC = Q_RET;
      #endif
    }

    //bookkeeping:
    dumpStats(Vstats[thrID], A_cur+V_cur, Ver ); //Qer
    data->Set[seq]->tuples[samp]->SquaredError = Ver*Ver;
    return gradient;
  }

  inline void offPolCorrUpdate(const Uint seq, const Uint samp, Real& Q_RET,
    Real& Q_OPC, const vector<Real>& output_hat, const Real rGamma) const
  {
    const Tuple*const _t=data->Set[seq]->tuples[samp]; //contains sOld, a
    const Tuple*const t_=data->Set[seq]->tuples[samp+1]; //contains r, sNew
    Q_RET = t_->r + rGamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = t_->r + rGamma*Q_OPC;
    const Real V_hat = output_hat[net_indices[0]];
    const Gaussian_policy pol_hat = prepare_policy(output_hat);
    //Used as target: target policy, target value
    const Quadratic_advantage adv_hat = prepare_advantage(output_hat,&pol_hat);
    //off policy stored action:
    const vector<Real> act = aInfo.getInvScaled(_t->a);//unbounded action space
    const Real actProbOnTarget = pol_hat.evalLogProbability(act);
    const Real actProbBehavior = Gaussian_policy::evalBehavior(act,_t->mu);
    const Real rho_hat = safeExp(actProbOnTarget-actProbBehavior);
    const Real c_hat = std::min((Real)1., std::pow(rho_hat, 1./nA));
    const Real A_hat = adv_hat.computeAdvantage(act);
    //prepare rolled Q with off policy corrections for next step:
    Q_RET = c_hat*ACER_LAMBDA*(Q_RET -A_hat -V_hat) +V_hat;
    //Q_OPC =     ACER_LAMBDA*(Q_OPC -A_hat -V_hat) +V_hat;
    Q_OPC = Q_RET;
  }

public:
  RACER(MPI_Comm comm, Environment*const env, Settings & settings);
  void select(const int agentId, const Agent& agent) override;

  void test();

  static Uint getnOutputs(const Uint NA)
  {
#if defined ACER_RELAX
    // I output V(s), P(s), pol(s), prec(s) (and variate)
    return 1+compute_nL(NA)+NA+NA;
#elif defined ACER_SAFE
    // I output V(s), P(s), pol(s), mu(s) (and variate)
    return 1+compute_nL(NA)+NA+NA;
#else //full formulation
    // I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
    return 1+compute_nL(NA)+NA+NA+NA;
#endif
  }
  static inline Uint compute_nL(const Uint NA)
  {
    return (NA*NA + NA)/2;
  }
};
