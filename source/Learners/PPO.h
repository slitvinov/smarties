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
#include "../Math/Discrete_advantage.h"
#include "../Network/Builder.h"
#define PPO_PENALKL
#define PPO_CLIPPED
//#define IGNORE_CRITIC
#define PPO_simpleSigma
#define PPO_learnDKLt

template<typename Policy_t, typename Action_t>
class PPO : public Learner_onPolicy
{
protected:
  const Uint nA = Policy_t::compute_nA(&aInfo);
  mutable vector<long double> valPenal, cntPenal;
  const Real lambda;
  const vector<Uint> pol_outputs, pol_indices;
  mutable std::atomic<Real> DKL_target;
  //tracks statistics about gradient, used for gradient clipping:
  StatsTracker* opcInfo;

  inline Policy_t prepare_policy(const Rvec& out,
    const Tuple*const t = nullptr) const {
    Policy_t pol(pol_indices, &aInfo, out);
    if(t not_eq nullptr) pol.prepare(t->a, t->mu);
    return pol;
  }

  void TrainBySequences(const Uint seq, const Uint thrID) const override {
    die("not allowed");
  }

  inline void updateDKL_target(const bool farPolSample, const Real DivKL) const
  {
    #ifdef PPO_learnDKLt
      //In absence of penalty term, it happens that within nEpochs most samples
      //are far-pol and therefore policy loss is 0. To keep samples on policy
      //we adapt DKL_target s.t. approx. 80% of samples are always near-Policy.
      //For most gym tasks with eta=1e-4 this results in ~0 penalty term.
      if(      farPolSample && DKL_target>DivKL) DKL_target = DKL_target*0.9995;
      else if(!farPolSample && DKL_target<DivKL) DKL_target = DKL_target*1.0001;
    #endif
  }

  void Train(const Uint seq, const Uint samp, const Uint thrID) const override
  {
    if(thrID==1)  profiler->stop_start("FWD");
    Sequence* const traj = data->Set[seq];
    const Real adv_est = traj->action_adv[samp], val_tgt = traj->Q_RET[samp];
    const Rvec MU = traj->tuples[samp]->mu;

    F[0]->prepare_one(traj, samp, thrID);
    const Rvec pol_cur = F[0]->forward(traj, samp, thrID);

    if(thrID==1)  profiler->stop_start("CMP");

    const Policy_t pol = prepare_policy(pol_cur, traj->tuples[samp]);
    const Real rho_cur = pol.sampImpWeight, DivKL = pol.kl_divergence(MU);
    Rvec sampleInfo {0, 0, 0, DivKL, rho_cur};
    const bool isFarPol = traj->isFarPolicyPPO(samp, rho_cur, CmaxPol);

    cntPenal[thrID+1]++;
    if(DivKL < DKL_target / 1.5) valPenal[thrID+1] -= valPenal[0]/2; //half
    if(DivKL > 1.5 * DKL_target) valPenal[thrID+1] += valPenal[0]; //double

    Real gain = rho_cur*adv_est;
    #ifdef PPO_CLIPPED
      if(adv_est > 0 && rho_cur > 1+CmaxPol) gain = 0;
      if(adv_est < 0 && rho_cur < 1-CmaxPol) gain = 0;
      updateDKL_target(isFarPol, DivKL);
    #endif

    F[1]->prepare_one(traj, samp, thrID);
    const Rvec val_cur = F[1]->forward(traj, samp, thrID);

    #ifdef PPO_PENALKL //*nonZero(gain)
      const Rvec policy_grad = pol.policy_grad(pol.sampAct, gain);
      const Rvec penal_grad = pol.div_kl_grad(MU, -valPenal[0]);
      const Rvec totalPolGrad = sum2Grads(penal_grad, policy_grad);
      for(Uint i=0; i<policy_grad.size(); i++) {
        sampleInfo[0] += policy_grad[i]*policy_grad[i];
        sampleInfo[1] +=  penal_grad[i]* penal_grad[i];
        sampleInfo[2] += policy_grad[i]* penal_grad[i];
      }
      sampleInfo[0] = std::sqrt(sampleInfo[0]);
      sampleInfo[1] = std::sqrt(sampleInfo[1]);
      sampleInfo[2] = sampleInfo[2]/(sampleInfo[1]+nnEPS);
    #else //we still learn the penal coef, for simplicity, but no effect
      const Rvec totalPolGrad = pol.policy_grad(pol.sampAct, gain);
      for(Uint i=0; i<totalPolGrad.size(); i++) {
        sampleInfo[0] += std::fabs(totalPolGrad[i]);
        sampleInfo[1] += 0; sampleInfo[2] += 0;
      }
    #endif

    Rvec grad(F[0]->nOutputs(), 0);
    pol.finalize_grad(totalPolGrad, grad);

    //bookkeeping:
    Vstats[thrID].dumpStats(val_cur[0], val_tgt - val_cur[0]);
    opcInfo->track_vector(sampleInfo, thrID);
    if(thrID==1)  profiler->stop_start("BCK");
    //if(!thrID) cout << "back pol" << endl;
    F[0]->backward(grad, samp, thrID);
    //if(!thrID) cout << "back val" << endl; //*(!isFarPol)
    F[1]->backward({(val_tgt - val_cur[0])*(!isFarPol)}, samp, thrID);
    F[0]->gradient(thrID);
    F[1]->gradient(thrID);
  }

public:
  PPO(Environment*const _env, Settings& _set, vector<Uint> pol_outs) :
    Learner_onPolicy(_env,_set), valPenal(nThreads+1,0), cntPenal(nThreads+1,0),
    lambda(_set.lambda), pol_outputs(pol_outs), pol_indices(count_indices(pol_outs)),
    DKL_target(_set.klDivConstraint) {
    opcInfo = new StatsTracker(5, "PPO", _set, 100);
    valPenal[0] = 1;
    //valPenal[0] = 1.;
  }

  //called by scheduler:
  void select(const Agent& agent) override
  {
    const int thrID= omp_get_thread_num();
    Sequence*const curr_seq = data->inProgress[agent.ID];
    data->add_state(agent);

    if(agent.Status < TERM_COMM ) { //non terminal state
      //Compute policy and value on most recent element of the sequence:
      const Rvec pol = F[0]->forward_agent(curr_seq, agent, thrID);
      const Rvec val = F[1]->forward_agent(curr_seq, agent, thrID);

      curr_seq->state_vals.push_back(val[0]);
      Policy_t policy = prepare_policy(pol);
      const Rvec MU = policy.getVector();
      const Action_t act = policy.finalize(bTrain, &generators[thrID], MU);
      agent.a->set(act);
      data->add_action(agent, MU);
    } else if( agent.Status == TRNC_COMM ) {
      const Rvec val = F[1]->forward_agent(curr_seq, agent, thrID);
      curr_seq->state_vals.push_back(val[0]);
    } else
      curr_seq->state_vals.push_back(0); // Assign value of term state to 0

    updatePPO(curr_seq);

    //advance counters of available data for training
    if(agent.Status >= TERM_COMM) data->terminate_seq(agent);
  }

  static inline Real annealDiscount(const Real targ, const Real orig,
    const Real t, const Real T=1e5) {
      // this function makes 1/(1-ret) linearly go from
      // 1/(1-orig) to 1/(1-targ) for t = 0:T. if t>=T it returns targ
    assert(targ>=orig);
    return t<T ? 1 -(1-orig)*(1-targ)/(1-targ +(t/T)*(targ-orig)) : targ;
  }
  void updatePPO(Sequence*const seq) const
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
    const Real R = data->scaledReward(seq,N-1);
    // delta_t = r_t+1 + gamma V(s_t+1) - V(s_t)  (pedix on r means r_t+1
    // received with transition to s_t+1, sometimes referred to as r_t)

    const Real delta = R +gamma*vSnew -vSold;
    seq->action_adv.push_back(0);
    seq->Q_RET.push_back(0);

    Real fac_lambda = 1, fac_gamma = 1;
    // If user selects gamma=.995 and lambda=0.97 as in Henderson2017
    // these will start at 0.99 and 0.95 (same as original) and be quickly
    // annealed upward in the first 1e5 steps.
    const Real rGamma  =  gamma>0.99? annealDiscount( gamma,.99,nStep) :  gamma;
    const Real rLambda = lambda>0.95? annealDiscount(lambda,.95,nStep) : lambda;
    // reward of i=0 is 0, because before any action
    // adv(0) is also 0, V(0) = V(s_0)
    for (int i=N-2; i>=0; i--) { //update all rewards before current step
      //will contain MC sum of returns:
      seq->Q_RET[i] += fac_gamma * R;
      #ifndef IGNORE_CRITIC
        seq->action_adv[i] += fac_lambda * delta;
      #else
        seq->action_adv[i] += fac_gamma * R;
      #endif
      fac_lambda *= rLambda*rGamma;
      fac_gamma *= rGamma;
    }
  }

  void getMetrics(ostringstream& buff) const {
    opcInfo->reduce_approx();
    {
      const Real v = valPenal[0];
      const int prec = std::fabs(v)>=10? 2 : (std::fabs(v)>=1? 3 : 4);
      buff<<" "<<std::setw(6)<<std::setprecision(prec)<<std::fixed<<valPenal[0];
    }
    {
      const Real v = opcInfo->instMean[0];
      const int p=std::fabs(v)>1e3?0:(std::fabs(v)>1e2?1:(std::fabs(v)>10?2:3));
      buff <<" " <<std::setw(6) <<std::setprecision(p) <<std::fixed <<v;
    }
    {
      const Real v = opcInfo->instMean[1];
      const int p=std::fabs(v)>1e3?0:(std::fabs(v)>1e2?1:(std::fabs(v)>10?2:3));
      buff <<" " <<std::setw(6) <<std::setprecision(p) <<std::fixed <<v;
    }
    {
      const Real v = opcInfo->instMean[2];
      const int p=std::fabs(v)>1e3?0:(std::fabs(v)>1e2?1:(std::fabs(v)>10?2:3));
      buff <<" " <<std::setw(6) <<std::setprecision(p) <<std::fixed <<v;
    }
    buff<<" "<<std::setw(6)<<std::setprecision(4)<<opcInfo->instMean[3];
    buff<<" "<<std::setw(6)<<std::setprecision(4)<<opcInfo->instMean[4];
    #ifdef PPO_learnDKLt
      buff<<" "<<std::setw(6)<<std::setprecision(4)<<DKL_target;
    #endif
  }
  void getHeaders(ostringstream& buff) const {
    buff <<"| beta | polG | penG | proj |  DKL | avgW "
    #ifdef PPO_learnDKLt
      "| DKLt "
    #endif
    ;
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
    const Real fac = learnR/cntPenal[0]; // learnRate*grad/N //
    cntPenal[0] = 0;
    for(Uint i=1; i<=nThreads; i++) {
        valPenal[0] += fac*valPenal[i];
        valPenal[i] = 0;
    }
    if(valPenal[0] <= nnEPS) valPenal[0] = nnEPS;
  }
};

class PPO_cont : public PPO<Gaussian_policy, Rvec >
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

  PPO_cont(Environment*const _env, Settings & _set) :
  PPO(_env, _set, count_pol_outputs(&_env->aI))
  {
    printf("Continuous-action PPO\n");
    #if 1
      if(input->net not_eq nullptr) {
        delete input->opt; input->opt = nullptr;
        delete input->net; input->net = nullptr;
      }
      Builder input_build(_set);
      bool bInputNet = false;
      input_build.addInput( input->nOutputs() );
      bInputNet = bInputNet || env->predefinedNetwork(input_build);
      bInputNet = bInputNet || predefinedNetwork(input_build);
      if(bInputNet) {
        Network* net = input_build.build(true);
        input->initializeNetwork(net, input_build.opt);
      }
    #endif
    F.push_back(new Approximator("policy", _set, input, data));
    F[0]->blockInpGrad = true;
    F.push_back(new Approximator("critic", _set, input, data));

    Builder build_val = F[1]->buildFromSettings(_set, {1} );

    #ifndef PPO_simpleSigma
      Rvec initBias;
      Gaussian_policy::setInitial_noStdev(&aInfo, initBias);
      Gaussian_policy::setInitial_Stdev(&aInfo, initBias, greedyEps);
      Builder build_pol = F[0]->buildFromSettings(_set, {2*aInfo.dim});
      build.setLastLayersBias(initBias);
    #else  //stddev params
      Builder build_pol = F[0]->buildFromSettings(_set,   {aInfo.dim});
      const Real initParam = Gaussian_policy::precision_inverse(greedyEps);
      build_pol.addParamLayer(aInfo.dim, "Linear", initParam);
    #endif
    F[0]->initializeNetwork(build_pol);

    _set.learnrate *= 3;
    F[1]->initializeNetwork(build_val);

    {  // TEST FINITE DIFFERENCES:
      Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
      std::normal_distribution<Real> dist(0, 1);
      for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
      for(Uint i=0;  i<mu.size(); i++) mu[i] = dist(generators[0]);
      for(Uint i=nA; i<mu.size(); i++) mu[i] = std::exp(mu[i]);

      Gaussian_policy pol = prepare_policy(output);
      Rvec act = pol.finalize(1, &generators[0], mu);
      pol.prepare(act, mu);
      pol.test(act, mu);
    }
  }
};

class PPO_disc : public PPO<Discrete_policy, Uint>
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
  PPO_disc(Environment*const _env, Settings& _set) :
  PPO(_env, _set, count_pol_outputs(&_env->aI))
  {
    printf("Discrete-action PPO\n");
    F.push_back(new Approximator("policy", _set, input, data));
    F.push_back(new Approximator("critic", _set, input, data));
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

printf("PPO: Built network with outputs: %s %s\n",
  print(net_indices).c_str(),print(net_outputs).c_str());
*/