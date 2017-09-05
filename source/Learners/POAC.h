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

class POAC : public Learner_utils
{
  //typedef Quadratic_advantage Advantage;
  typedef Diagonal_advantage Advantage;
  const Real truncation, DKL_target, DKL_hardmax, CmaxRet = 1, CmaxRho = 1;
  const Uint nA, nL;
  std::vector<std::mt19937>& generators;

  #if defined ACER_RELAX //  V(s),P(s),pol(s), precA(s), precQ(q)/penal
    vector<Uint> net_outputs = {1, nL,   nA,      nA,         2};
    vector<Uint> net_indices = {0,  1, 1+nL, 1+nL+nA, 1+nL+2*nA};
    const Uint QPrecID = net_indices[4], PenalID = net_indices[4]+1;
  #else // output            V(s),P(s),pol(s), mu(s),  precA(s), precQ(q)/penal
    vector<Uint> net_outputs = {1, nL,   nA,      nA,        nA,         2};
    vector<Uint> net_indices = {0,  1, 1+nL, 1+nL+nA, 1+nL+2*nA, 1+nL+3*nA};
    const Uint QPrecID = net_indices[5], PenalID = net_indices[5]+1;
  #endif

  #ifdef FEAT_CONTROL
    const ContinuousSignControl* task;
  #endif

  inline Gaussian_policy prepare_policy(const vector<Real>& out) const
  {
    #if defined ACER_RELAX
    return Gaussian_policy(net_indices[2], net_indices[3], nA, out);
    #else
    return Gaussian_policy(net_indices[2], net_indices[4], nA, out);
    #endif
  }
  inline Advantage prepare_advantage(const vector<Real>& out,
      const Gaussian_policy*const pol) const
  {
    #if defined ACER_RELAX
    return Advantage(net_indices[1], nA, nL, out, pol);
    #else
    return Advantage(net_indices[1],net_indices[3],nA,nL,out,pol);
    #endif
  }

  void Train_BPTT(const Uint seq, const Uint thrID) const override;
  void Train(const Uint seq, const Uint samp, const Uint thrID) const override;

  inline vector<Real> compute(const Uint seq, const Uint samp, Real& Q_RET,
    Real& Q_OPC, const vector<Real>& out_cur, const vector<Real>& out_hat,
    const Real rGamma, const Uint thrID) const
  {
    const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
    //const Tuple * const t_ =data->Set[seq]->tuples[samp+1]; //contains r, sNew
    const Real reward = data->standardized_reward(seq, samp+1);
    Q_RET = reward + rGamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = reward + rGamma*Q_OPC;
    //get everybody camera ready:
    const Real V_cur = out_cur[net_indices[0]]; //V_hat = out_hat[net_indices[0]];
    const Real Qprecision = out_cur[QPrecID], penalDKL = out_cur[PenalID];
    const Gaussian_policy pol_cur = prepare_policy(out_cur);
    const Gaussian_policy pol_hat = prepare_policy(out_hat);
    const Advantage adv_cur = prepare_advantage(out_cur, &pol_cur);
    const Real A_OPC = Q_OPC - V_cur;

    //off policy stored action and on-policy sample:
    const vector<Real> act = aInfo.getInvScaled(_t->a); //unbounded action space
    const Real actProbOnPolicy = pol_cur.evalLogProbability(act);
    const Real actProbBehavior = Gaussian_policy::evalBehavior(act,_t->mu);
    const Real rho_cur = min(MAX_IMPW,safeExp(actProbOnPolicy-actProbBehavior));
    const Real DivKL = pol_cur.kl_divergence_opp(&pol_hat);
    const Real A_cur = adv_cur.computeAdvantage(act);
    const Real Qer = Q_RET -A_cur -V_cur;

    //compute quantities needed for trunc import sampl with bias correction
    #ifdef ACER_TABC
      const vector<Real> pol = pol_cur.sample(&generators[thrID]);
      const Real polProbOnPolicy = pol_cur.evalLogProbability(pol);
      const Real polProbBehavior = Gaussian_policy::evalBehavior(pol,_t->mu);
      const Real rho_pol = safeExp(polProbOnPolicy-polProbBehavior);
      const Real A_pol = adv_cur.computeAdvantage(pol);
      const Real gain1 = A_OPC*min(rho_cur,truncation);
      const Real gain2 = A_pol*max(0.,1.-truncation/rho_pol);

      const vector<Real> gradAcer_1 = pol_cur.policy_grad(act, gain1);
      const vector<Real> gradAcer_2 = pol_cur.policy_grad(pol, gain2);
      const vector<Real> gradAcer = sum2Grads(gradAcer_1, gradAcer_2);
    #else
      //const Real gain1 = A_OPC * rho_cur;
      const Real gain1 = rho_cur>CmaxRho && A_OPC>0 ? CmaxRho*A_OPC : A_OPC*rho_cur;
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

    //trust region updating
    const vector<Real> penal_grad = pol_cur.div_kl_opp_grad(&pol_hat,-penalDKL);
    vector<Real> totalPolGrad = sum2Grads(penal_grad, policy_grad);

    #if 0
      const vector<Real> gradDivKL = pol_cur.div_kl_grad(&pol_hat);
      totalPolGrad = trust_region_update(totalPolGrad, gradDivKL, DKL_hardmax);
    #endif

    const Real Ver = Qer*std::min((Real)1, rho_cur);
    vector<Real> gradient(nOutputs,0);
    gradient[net_indices[0]]= Qer * Qprecision;

    //decrease precision if error is large
    //computed as \nabla_{Qprecision} Dkl (Q^RET_dist || Q_dist)
    gradient[QPrecID] = -.5 * (Qer * Qer - 1/Qprecision);
    //increase if DivKL is greater than Target
    //computed as \nabla_{penalDKL} (DivKL - DKL_target)^2
    //with rough approximation that DivKL/penalDKL = penalDKL
    //(distance increases if penalty term increases, similar to PPO )
    //gradient[PenalID] = 2*(DivKL - DKL_target)*penalDKL;
    gradient[PenalID] = 4*pow(DivKL - DKL_target,3)*penalDKL;

    //if ( thrID==1 ) printf("%u %u %u : %f %f DivKL:%f grad=[%f %f]\n", nOutputs, QPrecID, PenalID, Qprecision, penalDKL, DivKL, penalty_grad[0], policy_grad[0]);

    //adv_cur.grad(act, Qer, gradient, aInfo.bounded);
    adv_cur.grad(act, Qer * Qprecision, gradient, aInfo.bounded);
    pol_cur.finalize_grad(totalPolGrad, gradient, aInfo.bounded);

    //prepare Q with off policy corrections for next step:
    Q_RET = std::min((Real)1, rho_cur)*(Q_RET -A_cur -V_cur) +V_cur;
    Q_OPC = std::min(CmaxRet, rho_cur)*(Q_RET -A_cur -V_cur) +V_cur;

    //bookkeeping:
    dumpStats(Vstats[thrID], A_cur+V_cur, Qer ); //Ver
    data->Set[seq]->tuples[samp]->SquaredError = Ver*Ver;
    return gradient;
  }

  inline void offPolCorrUpdate(const Uint seq, const Uint samp, Real& Q_RET,
    Real& Q_OPC, const vector<Real>& output_hat, const Real rGamma) const
  {
    const Tuple * const _t = data->Set[seq]->tuples[samp]; //contains sOld, a
    //const Tuple * const t_ = data->Set[seq]->tuples[samp+1];//contains r, sNew
    const Real reward = data->standardized_reward(seq,samp+1);
    Q_RET = reward + rGamma*Q_RET; //if k==ndata-2 then this is r_end
    Q_OPC = reward + rGamma*Q_OPC;
    const Real V_hat = output_hat[net_indices[0]];
    const Gaussian_policy pol_hat = prepare_policy(output_hat);
    //Used as target: target policy, target value
    const Advantage adv_hat = prepare_advantage(output_hat, &pol_hat);
    //off policy stored action:
    const vector<Real> act = aInfo.getInvScaled(_t->a);//unbounded action space
    const Real actProbOnTarget = pol_hat.evalLogProbability(act);
    const Real actProbBehavior = Gaussian_policy::evalBehavior(act,_t->mu);
    const Real C = safeExp(actProbOnTarget-actProbBehavior);
    const Real A_hat = adv_hat.computeAdvantage(act);
    //prepare rolled Q with off policy corrections for next step:
    Q_RET = std::min((Real)1, C)*(Q_RET -A_hat -V_hat) +V_hat;
    Q_OPC = std::min(CmaxRet, C)*(Q_RET -A_hat -V_hat) +V_hat;
  }

  void myBuildNetwork(Network*& _net , Optimizer*& _opt,
      const vector<Uint> nouts, Settings & settings);
public:
  POAC(MPI_Comm comm, Environment*const env, Settings & settings);

  void select(const int agentId, const Agent& agent) override;

  void test();
  void processStats() override;
  static Uint getnOutputs(const Uint NA)
  {
    #if defined ACER_RELAX
      // I output V(s), P(s), pol(s), prec(s) (and variate)
      return 1 +Advantage::compute_nL(NA) +NA +NA +2;
    #else //full formulation
      // I output V(s), P(s), pol(s), prec(s), mu(s) (and variate)
      return 1 +Advantage::compute_nL(NA) +NA +NA +NA +2;
    #endif
  }
};
