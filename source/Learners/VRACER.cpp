//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Network/Builder.h"

#include "VRACER.h"

#include "../Math/Gaussian_mixture.h"
#include "../Math/Gaussian_policy.h"
#include "../Math/Discrete_policy.h"

#define DACER_simpleSigma
#define DACER_singleNet

template<typename Policy_t>
static inline Policy_t prepare_policy(const Rvec& O, const ActionInfo*const aI,
  const vector<Uint>& pol_indices, const Tuple*const t = nullptr) {
  Policy_t pol(pol_indices, aI, O);
  if(t not_eq nullptr) pol.prepare(t->a, t->mu);
  return pol;
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::TrainBySequences(const Uint seq,
  const Uint wID, const Uint bID, const Uint thrID) const {
  die("");
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::Train(const Uint seq, const Uint t,
  const Uint wID, const Uint bID, const Uint thrID) const
{
  Sequence* const S = data->get(seq);
  assert(t+1 < S->tuples.size());

  if(thrID==0) profiler->stop_start("FWD");
  F[0]->prepare_one(S, t, thrID, wID); // prepare thread workspace
  const Rvec out = F[0]->forward(t, thrID); // network compute

  #ifdef DACER_singleNet
    static constexpr int valNetID = 0;
    const Rvec& val = out;
  #else
    static constexpr int valNetID = 1;
    F[1]->prepare_one(S, t, thrID, wID); // prepare thread workspace
    const Rvec val = F[1]->forward(t, thrID); // network compute
  #endif

  if ( wID == 0 and S->isTruncated(t+1) ) {
    const Rvec nxt = F[valNetID]->forward(t+1, thrID);
    S->setStateValue(t+1, nxt[VsID]);
  }

  if(thrID==0)  profiler->stop_start("CMP");

  const Policy_t P = prepare_policy<Policy_t>(out, &aInfo, pol_start, S->tuples[t]);
  // check whether importance weight is in 1/Cmax < c < Cmax
  const Real R = data->scaledReward(S, t+1), W = P.sampImpWeight;
  const Real Vcur = val[0], Vnxt = S->state_vals[t+1], Dnxt = S->Q_RET[t+1];
  const bool isOff = S->isFarPolicy(t, W, CmaxRet, CinvRet);
  const Real A_RET = R + gamma * (Dnxt + Vnxt) - Vcur;
  const Real D_RET = std::min((Real)1, W) * A_RET;

  if( wID == 0 ) {
    S->setStateValue(t, Vcur); S->setRetrace(t, D_RET);
    S->setMseDklImpw(t, D_RET*D_RET, P.sampKLdiv, W, CmaxRet, CinvRet);
  }

  if(ESpopSize>1) {
    advs[bID][wID] = A_RET;
    dkls[bID][wID] = P.sampKLdiv;
    rhos[bID][wID] = P.sampImpWeight;
    if( wID == 0 )
      trainInfo->log(Vcur, D_RET, {std::pow(D_RET-S->Q_RET[t],2), W}, thrID);
  }
  else
  {
    assert(wID == 0);
    Rvec G = Rvec(F[0]->nOutputs(), 0);
    if(isOff) P.finalize_grad(P.div_kl_grad(S->tuples[t]->mu, beta-1), G);
    else {
     const Rvec G1 = P.policy_grad(P.sampAct, A_RET * W);
     const Rvec G2 = P.div_kl_grad(S->tuples[t]->mu, -1);
     P.finalize_grad(weightSum2Grads(G1, G2, beta), G);
     trainInfo->log(Vcur,D_RET,G1,G2, {std::pow(D_RET-S->Q_RET[t],2),W}, thrID);
    }

    if(thrID==0) profiler->stop_start("BCK");
    #ifdef DACER_singleNet
      assert(std::fabs(G[0])<1e-16); G[0] = beta * D_RET;
    #else
      F[1]->backward( Rvec(1, beta * D_RET), t, thrID);
      F[1]->gradient(thrID);  // backprop
    #endif
    F[0]->backward(G, t, thrID);
    F[0]->gradient(thrID);  // backprop
  }
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::select(Agent& agent)
{
  Sequence* const traj = data_get->get(agent.ID);
  data_get->add_state(agent);
  F[0]->prepare_agent(traj, agent);

  #ifdef DACER_singleNet
    static constexpr int valNetID = 0;
  #else
    static constexpr int valNetID = 1;
    F[1]->prepare_agent(traj, agent);
  #endif

  if( agent.Status < TERM_COMM ) // not last of a sequence
  {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    Rvec output = F[0]->forward_agent(agent);
    #ifdef DACER_singleNet
      const Rvec& value = output;
    #else
      Rvec value = F[1]->forward_agent(agent);
    #endif
    Policy_t pol = prepare_policy<Policy_t>(output, &aInfo, pol_start);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    auto act = pol.finalize(explNoise>0, &generators[nThreads+agent.ID], mu);

    traj->state_vals.push_back(value[0]);
    agent.act(act);
    data_get->add_action(agent, mu);
  }
  else
  {
    if( agent.Status == TRNC_COMM ) {
      Rvec output = F[valNetID]->forward_agent(agent);
      traj->state_vals.push_back(output[0]);
    } else traj->state_vals.push_back(0); //value of term state is 0

    assert(traj->tuples.size() == traj->state_vals.size());
    assert(traj->Q_RET.size() == 0);
    const Uint N = traj->tuples.size();
    // compute initial Qret for whole trajectory
    //within Retrace, we use the state_vals vector to write the Q retrace values
    traj->Q_RET.resize(N, 0);
    traj->Q_RET[N-1] = 0; //both if truncated or not, last delta is zero
    //update all q_ret before terminal step.
    for(Uint i=N-1; i>0; i--) updateVret(traj, i-1, 1);

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::prepareGradient()
{
  if(updateComplete and ESpopSize>1)
  {
    profiler->stop_start("LOSS");
    std::vector<Real> aR(batchSize, 0), aA(batchSize, 0);
    #if 1
     #pragma omp parallel for schedule(static)
     for (Uint b=0; b<batchSize; b++) {
       for(Uint w=0; w<ESpopSize; w++) { aR[b]+=rhos[b][w]; aA[b]+=advs[b][w]; }
       aR[b] /= ESpopSize; aA[b] /= ESpopSize;
     }
    #else
     for(Uint b=0; b<batchSize; b++) { aR[b] = rhos[b][0]; aA[b] = advs[b][0]; }
    #endif

    const auto isFar = [&](const Real&W) {return W >= CmaxRet || W <= CinvRet;};
    #pragma omp parallel for schedule(static)
    for (Uint w=0; w<ESpopSize; w++)
    for (Uint b=0; b<batchSize; b++) {
      const Real clipR = std::max(CinvRet, std::min(rhos[b][w], CmaxRet));
      const Real clipA = isFar(rhos[b][w]) ? aA[b] : advs[b][w];
      const Real costAdv = - beta * clipR * aA[b]; //minus: to maximize pol adv
      const Real costVal = beta * std::pow(std::min((Real)1, aR[b]) * clipA, 2);
      const Real costDkl = (1-beta) * dkls[b][w];
      #ifdef DACER_singleNet
        F[0]->losses[w] += alpha * (costAdv + costDkl) + (1-alpha) * costVal;
      #else
        F[0]->losses[w] += costAdv + costDkl;
        F[1]->losses[w] += costVal;
      #endif
    }
    F[0]->nAddedGradients = ESpopSize * batchSize;
    #ifndef DACER_singleNet
      F[1]->nAddedGradients = ESpopSize * batchSize;
    #endif
  }

  Learner_offPolicy::prepareGradient();

  if(updateToApply)
  {
    profiler->stop_start("QRET");
    debugL("Update Retrace est. for episodes samples in prev. grad update");
    // placed here because this happens right after update is computed
    // this can happen before prune and before workers are joined
    const std::vector<Uint>& sampled = data->listSampled();
    const Uint setSize = sampled.size();
    #pragma omp parallel for schedule(dynamic)
    for(Uint i = 0; i < setSize; i++) {
      Sequence * const S = data->get(sampled[i]);
      for(int j=S->just_sampled; j>=0; j--) updateVret(S,j,S->offPolicImpW[j]);
    }
  }
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::initializeLearner()
{
  Learner_offPolicy::initializeLearner();

  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) Retrace values for all obss
  // seen before this point.
  debugL("Rescale Retrace est. after gathering initial dataset");
  const Uint setSize = data->readNSeq();
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < setSize; i++) {
    Sequence* const traj = data->get(i);
    const int N = traj->ndata(); traj->setRetrace(N, 0);
    for(Uint j=N; j>0; j--) updateVret(traj, j-1, 1);
  }
}

///////////////////////////////////////////////////////////////////////////////

template<> vector<Uint> VRACER<Discrete_policy, Uint>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, aI->maxLabel};
}
template<> vector<Uint> VRACER<Discrete_policy, Uint>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return vector<Uint>{indices[1]};
  #else
    return vector<Uint>{indices[0]};
  #endif
}
template<> Uint VRACER<Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo*const aI) { return aI->maxLabel; }

template<> VRACER<Discrete_policy, Uint>::
VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI))
{
  printf("Discrete-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
  setupNet();
}

/////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////

template<> vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, NEXPERTS, NEXPERTS*aI->dim, NEXPERTS*aI->dim};
}
template<> vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return vector<Uint>{indices[1], indices[2], indices[3]};
  #else
    return vector<Uint>{indices[0], indices[1], indices[2]};
  #endif
}
template<> Uint VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
getnDimPolicy(const ActionInfo*const aI) { return NEXPERTS*(1 +2*aI->dim); }

template<> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI))
{
  printf("Mixture-of-experts continuous-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
  setupNet();

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
    for(Uint i=0; i<mu.size(); i++) mu[i] = dist(generators[0]);
    Real norm = 0;
    for(Uint i=0; i<NEXPERTS; i++) {
      mu[i] = std::exp(mu[i]);
      norm += mu[i];
    }
    for(Uint i=0; i<NEXPERTS; i++) mu[i] = mu[i]/norm;
    for(Uint i=NEXPERTS*(1+nA);i<NEXPERTS*(1+2*nA);i++) mu[i]=std::exp(mu[i]);

    Gaussian_mixture<NEXPERTS> pol = prepare_policy<Gaussian_mixture<NEXPERTS>>(output, &aInfo, pol_start);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template<> vector<Uint> VRACER<Gaussian_policy, Rvec>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, aI->dim, aI->dim};
}
template<> vector<Uint> VRACER<Gaussian_policy, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return vector<Uint>{indices[1], indices[2]};
  #else
    return vector<Uint>{indices[0], indices[1]};
  #endif
}
template<> Uint VRACER<Gaussian_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI) { return 2*aI->dim; }

template<> VRACER<Gaussian_policy, Rvec>::
VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI)) {
  printf("Gaussian continuous-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
  setupNet();

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);

    for(Uint i=0; i<mu.size(); i++) mu[i] = dist(generators[0]);
    for(Uint i=0; i<nA; i++) mu[i+nA] = std::exp(0.5*mu[i+nA] -1);
    for(Uint i=0; i<nA; i++) output[1+i] = mu[i] + dist(generators[0])*mu[i+nA];
    for(Uint i=0; i<nA; i++) {
      const Real S = mu[i+nA];//*mu[i+nA];
      output[1+i+nA] = (S*S -.25)/S + .1*dist(generators[0]);
    }

    Gaussian_policy pol = prepare_policy<Gaussian_policy>(output, &aInfo, pol_start);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

///////////////////////////////////////////////////////////////////////////////

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::setupNet()
{
  const std::type_info& actT = typeid(Action_t);
  const std::type_info& vecT = typeid(Rvec);
  const bool isContinuous = actT.hash_code() == vecT.hash_code();

  vector<Uint> nouts = count_outputs(&aInfo);

  #ifdef DACER_singleNet // state value is approximated by an other net
    F.push_back(new Approximator("net", settings, input, data));
  #else
    nout.erase( nout.begin() );
    F.push_back(new Approximator("policy", settings, input, data));
    F[0]->blockInpGrad = true; // this line must happen b4 initialize
    F.push_back(new Approximator("critic", settings, input, data));
    // make value network:
    Builder build_val = F[1]->buildFromSettings(settings, 1);
    F[1]->initializeNetwork(build_val);
  #endif

  #ifdef DACER_simpleSigma // variance not dependent on state
    const Uint varianceSize = nouts.back();
    if(isContinuous) nouts.pop_back();
  #endif

  Builder build = F[0]->buildFromSettings(settings, nouts);

  if(isContinuous)
  {
    #ifdef DACER_singleNet
      Rvec initBias = Rvec(1, 0);
    #else
      Rvec initBias = Rvec(0, 0); // no state val here
    #endif
    Policy_t::setInitial_noStdev(&aInfo, initBias);

    #ifdef DACER_simpleSigma // sigma not linked to state: param output
      build.setLastLayersBias(initBias);
      #ifdef EXTRACT_COVAR
        Real initParam = noiseMap_inverse(explNoise*explNoise);
      #else
        Real initParam = noiseMap_inverse(explNoise);
      #endif
      build.addParamLayer(varianceSize, "Linear", initParam);
    #else
      Policy_t::setInitial_Stdev(&aInfo, initBias, explNoise);
      build.setLastLayersBias(initBias);
    #endif
  }

  // construct policy net:
  F[0]->initializeNetwork(build);
  trainInfo = new TrainData("v-racer", settings,ESpopSize<2,"| dAdv | avgW ",2);
}

///////////////////////////////////////////////////////////////////////////////

template class VRACER<Discrete_policy, Uint>;
template class VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;
template class VRACER<Gaussian_policy, Rvec>;
