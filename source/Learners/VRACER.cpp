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

  if ( wID == 0 and S->isTruncated(t+1) ) {
    const Rvec nxt = F[0]->forward(t+1, thrID);
    S->setStateValue(t+1, nxt[VsID]);
  }

  if(thrID==0)  profiler->stop_start("CMP");

  const Policy_t P = prepare_policy<Policy_t>(out, &aInfo, pol_start, S->tuples[t]);
  // check whether importance weight is in 1/Cmax < c < Cmax
  const Real R = data->scaledReward(S, t+1), W = P.sampImpWeight;
  const Real Vcur = out[0], Vnxt = S->state_vals[t+1], Dnxt = S->Q_RET[t+1];
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
     const Rvec G2  = P.div_kl_grad(S->tuples[t]->mu, -1);
     P.finalize_grad(weightSum2Grads(G1, G2, beta), G);
     trainInfo->log(Vcur,D_RET,G1,G2, {std::pow(D_RET-S->Q_RET[t],2),W}, thrID);
    }
    if(thrID==0) profiler->stop_start("BCK");
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

  if( agent.Status < TERM_COMM ) // not last of a sequence
  {
    //Compute policy and value on most recent element of the sequence. If RNN
    // recurrent connection from last call from same agent will be reused
    Rvec output = F[0]->forward_agent(agent);
    Policy_t pol = prepare_policy<Policy_t>(output, &aInfo, pol_start);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    auto act = pol.finalize(explNoise>0, &generators[nThreads+agent.ID], mu);

    #if 0 // add and update temporally correlated noise
      act = pol.updateOrUhState(OrUhState[agent.ID], mu, act, iter());
    #endif
    //if(nStep) cout << "pol:" << print(mu) << endl;

    traj->state_vals.push_back(output[VsID]);
    agent.act(act);
    data_get->add_action(agent, mu);

    #ifndef NDEBUG
      //Policy_t dbg = prepare_policy<Policy_t>(output, &aInfo, pol_start);
      //dbg.prepare(traj->tuples.back()->a, traj->tuples.back()->mu);
      //const double err = fabs(dbg.sampImpWeight-1);
      //if(err>1e-10) _die("Imp W err %20.20e", err);
    #endif
  }
  else
  {
    if( agent.Status == TRNC_COMM ) {
      Rvec output = F[0]->forward_agent(agent);
      traj->state_vals.push_back(output[VsID]);
    } else
      traj->state_vals.push_back(0); //value of terminal state is 0

    writeOnPolRetrace(traj); // compute initial Qret for whole trajectory
    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::writeOnPolRetrace(Sequence*const seq) const
{
  assert(seq->tuples.size() == seq->state_vals.size());
  assert(seq->Q_RET.size() == 0);
  const Uint N = seq->tuples.size();
  //within Retrace, we use the state_vals vector to write the Q retrace values
  seq->Q_RET.resize(N, 0); //both if truncated or not, terminal delta is zero
  //update all q_ret before terminal step
  for (Uint i=N-1; i>0; i--) updateVret(seq, i-1, 1);
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::prepareGradient()
{
  if(updateComplete and ESpopSize>1)
  {
    profiler->stop_start("LOSS");
    std::vector<Real> aR(batchSize, 0), aA(batchSize, 0);
    #pragma omp parallel for schedule(static)
    for (Uint b=0; b<batchSize; b++) {
      for (Uint w=0; w<ESpopSize; w++) { aR[b]+=rhos[b][w]; aA[b]+=advs[b][w]; }
      aR[b] /= ESpopSize; aA[b] /= ESpopSize;
    }

    const auto isFar = [&](const Real&W) {return W >= CmaxRet || W <= CinvRet;};
    #pragma omp parallel for schedule(static)
    for (Uint w=0; w<ESpopSize; w++)
    for (Uint b=0; b<batchSize; b++) {
      const Real clipR = std::max(CinvRet, std::min(rhos[b][w], CmaxRet));
      const Real clipA = isFar(rhos[b][w]) ? aA[b] : advs[b][w];
      const Real rAdv = - clipR * aA[b]; //minus: to maximize advantage
      const Real dVret = std::pow( std::min((Real)1, aR[b]) * clipA, 2);
      const Real L = alpha*(beta*rAdv +(1-beta)*dkls[b][w]) + dVret*(1-alpha);
      F[0]->losses[w] += L;
    }
    F[0]->nAddedGradients = ESpopSize * batchSize;
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

  Learner_offPolicy::initializeLearner();
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
  return vector<Uint>{indices[1]};
}
template<> Uint VRACER<Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo*const aI) {
  return aI->maxLabel;
}

template<> VRACER<Discrete_policy, Uint>::
VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI)) {
  printf("Discrete-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());

  F.push_back(new Approximator("net", _set, input, data));
  vector<Uint> nouts{1, nA};
  Builder build = F[0]->buildFromSettings(_set, nouts);
  F[0]->initializeNetwork(build);

  trainInfo=new TrainData("v-racer", _set, ESpopSize<2, "| dAdv | avgW ", 2);
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template<> vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, NEXPERTS, NEXPERTS*aI->dim, NEXPERTS*aI->dim};
}
template<> vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1], indices[2], indices[3]};
}
template<> Uint VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
getnDimPolicy(const ActionInfo*const aI) {
  return NEXPERTS*(1 +2*aI->dim);
}

template<> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI)) {
  printf("Mixture-of-experts continuous-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());

  F.push_back(new Approximator("net", _set, input, data));
  vector<Uint> nouts{1, NEXPERTS, NEXPERTS * nA};
  #ifndef DACER_simpleSigma // network outputs also sigmas
    nouts.push_back(NEXPERTS * nA);
  #endif

  Builder build = F[0]->buildFromSettings(_set, nouts);
  Rvec initBias(1, 0); // state
  Gaussian_mixture<NEXPERTS>::setInitial_noStdev(&aInfo, initBias);
  #ifdef DACER_simpleSigma // sigma not linked to network: parametric output
    build.setLastLayersBias(initBias);
    #ifdef EXTRACT_COVAR
      Real initParam = noiseMap_inverse(explNoise*explNoise);
    #else
      Real initParam = noiseMap_inverse(explNoise);
    #endif
    build.addParamLayer(NEXPERTS * nA, "Linear", initParam);
  #else
    Gaussian_mixture<NEXPERTS>::setInitial_Stdev(&aInfo, initBias, explNoise);
    build.setLastLayersBias(initBias);
  #endif
  F[0]->initializeNetwork(build);

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

  trainInfo=new TrainData("v-racer", _set, ESpopSize<2, "| dAdv | avgW ", 2);
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
  return vector<Uint>{indices[1], indices[2]};
}
template<> Uint VRACER<Gaussian_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI) {
  return 2*aI->dim;
}

template<> VRACER<Gaussian_policy, Rvec>::
VRACER(Environment*const _env, Settings& _set): Learner_offPolicy(_env,_set),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI)) {
  printf("Gaussian continuous-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());

  F.push_back(new Approximator("net", _set, input, data));
  vector<Uint> nouts{1, nA};
  #ifndef DACER_simpleSigma // network outputs also sigmas
    nouts.push_back(nA);
  #endif

  Builder build = F[0]->buildFromSettings(_set, nouts);
  Rvec initBias(1, 0); // state
  Gaussian_policy::setInitial_noStdev(&aInfo, initBias);
  #ifdef DACER_simpleSigma // sigma not linked to network: parametric output
    build.setLastLayersBias(initBias);
    #ifdef EXTRACT_COVAR
      Real initParam = noiseMap_inverse(explNoise*explNoise);
    #else
      Real initParam = noiseMap_inverse(explNoise);
    #endif
    build.addParamLayer(nA, "Linear", initParam);
  #else
    Gaussian_policy::setInitial_Stdev(&aInfo, initBias, explNoise);
    build.setLastLayersBias(initBias);
  #endif
  F[0]->initializeNetwork(build);

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

  trainInfo=new TrainData("v-racer", _set, ESpopSize<2, "| dAdv | avgW ", 2);
}

template class VRACER<Discrete_policy, Uint>;
template class VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;
template class VRACER<Gaussian_policy, Rvec>;
