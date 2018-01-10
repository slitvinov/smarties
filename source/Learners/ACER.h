/*
 *  NAF.h
 *  rl
 *
 *  Created by Guido Novati on 16.07.15.
 *  Copyright 2013 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "../Network/Builder.h"
#include "Learner_offPolicy.h"
#include "../Math/FeatureControlTasks.h"
#include "../Math/Quadratic_advantage.h"

#ifndef NEXPERTS
#define NEXPERTS 1
#warning "Using Mixture_advantage with 1 expert"
#endif

#include "../Math/Discrete_policy.h"
#include "RACER.cpp"
#define simpleSigma

class ACER : public Learner_offPolicy
{
 protected:
  using Policy_t = Gaussian_policy;
  using Action_t = vector<Real>;
  const Uint nA = Policy_t::compute_nA(&aInfo);

  const Real CmaxPol, delta;
  const vector<Uint> net_outputs, net_indices, pol_start;

  StatsTracker* opcInfo;

  inline Policy_t prepare_policy(const vector<Real>& out) const {
    return Policy_t(pol_start, &aInfo, out);
  }
  inline Policy_t* new_policy(const vector<Real>& out) const {
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
  ACER(Environment*const _env, Settings& sett, vector<Uint> net_outs,
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
};

class ACER_cont : public ACER<Quadratic_advantage, Gaussian_policy, vector<Real> >
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
  static Uint getnDimPolicy(const ActionInfo*const aI)
  {
    return 2*aI->dim;
  }

  RACER_cont(Environment*const _env, Settings& settings) :
  RACER(_env, settings, count_outputs(_env->aI), count_pol_starts(_env->aI), count_adv_starts(_env->aI) )
  {
    printf("Continuous-action RACER: Built network with outputs: %s %s\n",
      print(net_indices).c_str(),print(net_outputs).c_str());
    F.push_back(new Approximator("net", settings, input, data));
    vector<Uint> nouts{1, nL, nA};
    #ifndef simpleSigma
      nouts.push_back(nA);
    #endif
    Builder build = F[0]->buildFromSettings(settings, nouts);
    #ifdef simpleSigma
      build.addParamLayer(nA, "Linear", -2*std::log(greedyEps));
    #endif
    //add klDiv penalty coefficient layer, and stdv of Q distribution
    build.addParamLayer(2, "Exp", 1/settings.klDivConstraint);
    F[0]->initializeNetwork(build);
  }
};
