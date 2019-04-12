//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Network/Builder.h"
#include "../Math/Gaussian_policy.h"
#include "../Math/Discrete_policy.h"

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::updatePPO(Sequence*const seq) const
{
  assert(seq->tuples.size());
  assert(seq->tuples.size() == seq->state_vals.size());

  //this is only triggered by t = 0 (or truncated trajectories)
  // at t=0 we do not have a reward, and we cannot compute delta
  //(if policy was updated after prev action we treat next state as initial)
  if(seq->state_vals.size() < 2)  return;
  assert(seq->tuples.size() == 2+seq->Q_RET.size());
  assert(seq->tuples.size() == 2+seq->action_adv.size());

  const Uint N = seq->tuples.size();
  const Fval vSold = seq->state_vals[N-2], vSnew = seq->state_vals[N-1];
  const Fval R = data->scaledReward(seq, N-1);
  // delta_t = r_t+1 + gamma V(s_t+1) - V(s_t)  (pedix on r means r_t+1
  // received with transition to s_t+1, sometimes referred to as r_t)

  const Fval delta = R +(Fval)gamma*vSnew -vSold;
  seq->action_adv.push_back(0);
  seq->Q_RET.push_back(0);

  Fval fac_lambda = 1, fac_gamma = 1;
  // If user selects gamma=.995 and lambda=0.97 as in Henderson2017
  // these will start at 0.99 and 0.95 (same as original) and be quickly
  // annealed upward in the first 1e5 steps.
  const Fval rGamma  =  gamma>.99? annealDiscount( gamma,.99,_nStep) :  gamma;
  const Fval rLambda = lambda>.95? annealDiscount(lambda,.95,_nStep) : lambda;
  // reward of i=0 is 0, because before any action
  // adv(0) is also 0, V(0) = V(s_0)
  for (int i=N-2; i>=0; i--) { //update all rewards before current step
    //will contain MC sum of returns:
    seq->Q_RET[i] += fac_gamma * R;
    //#ifndef IGNORE_CRITIC
      seq->action_adv[i] += fac_lambda * delta;
    //#else
    //  seq->action_adv[i] += fac_gamma * R;
    //#endif
    fac_lambda *= rLambda*rGamma;
    fac_gamma *= rGamma;
  }
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::initializeLearner()
{
  const Uint currStep = nStep();
  if ( currStep > 0 ) {
    warn("Skipping initialization for restartd learner.");
    return;
  }

  profiler->stop_start("PRE");
  debugL("Compute state/rewards stats from the replay memory");
  data_proc->updateRewardsStats(1, 1, true);
  if( learn_rank == 0 )
    printf("Initial reward std:%e\n",1/data->scaledReward(1));

  // Rewards second moment is computed right before actual training begins
  // therefore we need to recompute (rescaled) GAE and MC cumulative rewards
  // This assumes V(s) is initialized small, so we just rescale by std(rew)
  debugL("Rescale GAE est. after gathering initial dataset");
  // placed here because on 1st step we just computed first rewards statistics

  const Uint setSize = data->readNSeq();
  const Fval invstdR = data->scaledReward(1);
  #pragma omp parallel for schedule(dynamic)
  for(Uint i = 0; i < setSize; i++) {
    assert(data->get(i)->ndata()>=1);
    assert(data->get(i)->action_adv.size() == data->get(i)->ndata());
    assert(data->get(i)->Q_RET.size()      == data->get(i)->ndata());
    assert(data->get(i)->state_vals.size() == data->get(i)->ndata()+1);
    for (Uint j=data->get(i)->ndata()-1; j>0; j--) {
      data->get(i)->action_adv[j] *= invstdR;
      data->get(i)->Q_RET[j] *= invstdR;
    }
  }

  const Uint todoSize = data_get->nInProgress();
  for(Uint i = 0; i < todoSize; i++) {
    if(data_get->get(i)->tuples.size() <= 1) continue;
    for (Uint j=data_get->get(i)->ndata()-1; j>0; j--) {
      data_get->get(i)->action_adv[j] *= invstdR;
      data_get->get(i)->Q_RET[j] *= invstdR;
    }
  }
}

///////////////////////////////////////////////////////////////////////
/////////// TEMPLATE SPECIALIZATION FOR CONTINUOUS ACTIONS ////////////
///////////////////////////////////////////////////////////////////////
template<>
vector<Uint> PPO_contAct::count_pol_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{aI->dim, aI->dim};
}
template<>
vector<Uint> PPO_contAct::count_pol_starts(const ActionInfo*const aI)
{
  const vector<Uint> indices = count_indices(count_pol_outputs(aI));
  return vector<Uint>{indices[0], indices[1]};
}
template<>
Uint PPO_contAct::getnDimPolicy(const ActionInfo*const aI)
{
  return 2*aI->dim; // policy dimension is mean and diag covariance
}

template<> PPO_contAct::PPO(Environment*const E, Settings&S) :
Learner(E, S), pol_outputs(count_pol_outputs(&E->aI))
{
  #ifdef PPO_learnDKLt
   trainInfo = new TrainData("PPO", S, 1, "| beta |  DKL | avgW | DKLt ", 4);
  #else
   trainInfo = new TrainData("PPO", S, 1, "| beta |  DKL | avgW ", 3);
  #endif
  valPenal[0] = 1;

  printf("Continuous-action PPO\n");
  #if 0 // shared input layers
    createSharedEncoder();
  #endif
  F.push_back(new Approximator("policy", S, input, data));
  F[0]->blockInpGrad = true;
  F.push_back(new Approximator("critic", S, input, data));

  Builder build_val = F[1]->buildFromSettings(S, {1} );

  #ifndef PPO_simpleSigma
    Rvec initBias;
    Gaussian_policy::setInitial_noStdev(&aInfo, initBias);
    Gaussian_policy::setInitial_Stdev(&aInfo, initBias, explNoise);
    Builder build_pol = F[0]->buildFromSettings(S, {2*aInfo.dim});
    build.setLastLayersBias(initBias);
  #else  //stddev params
    Builder build_pol = F[0]->buildFromSettings(S,   {aInfo.dim});
    #ifdef EXTRACT_COVAR
      Real initParam = noiseMap_inverse(explNoise*explNoise);
    #else
      Real initParam = noiseMap_inverse(explNoise);
    #endif
    build_pol.addParamLayer(aInfo.dim, "Linear", initParam);
  #endif
  F[0]->initializeNetwork(build_pol);

  S.learnrate *= 3; // for shared input layers
  F[1]->initializeNetwork(build_val);
  S.learnrate /= 3;

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<output.size(); i++) output[i] = dist(generators[0]);
    for(Uint i=0;  i<mu.size(); i++) mu[i] = dist(generators[0]);
    for(Uint i=nA; i<mu.size(); i++) mu[i] = std::exp(mu[i]);

    Gaussian_policy pol = prepare_policy<Gaussian_policy>(output, &aInfo, pol_indices);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

///////////////////////////////////////////////////////////////////////
//////////// TEMPLATE SPECIALIZATION FOR DISCRETE ACTIONS /////////////
///////////////////////////////////////////////////////////////////////
template<>
vector<Uint> PPO_discAct::count_pol_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{aI->maxLabel};
}
template<>
vector<Uint> PPO_discAct::count_pol_starts(const ActionInfo*const aI)
{
  const vector<Uint> indices = count_indices(count_pol_outputs(aI));
  return vector<Uint>{indices[0]};
}
template<>
Uint PPO_discAct::getnDimPolicy(const ActionInfo*const aI)
{
  return aI->maxLabel;
}

template<> PPO_discAct::PPO(Environment*const E, Settings&S) :
Learner(E,S), pol_outputs(count_pol_outputs(&E->aI))
{
  #ifdef PPO_learnDKLt
    trainInfo = new TrainData("PPO", S, 1,"| beta |  DKL | avgW | DKLt ",4);
  #else
    trainInfo = new TrainData("PPO", S, 1,"| beta |  DKL | avgW ", 3);
  #endif
  valPenal[0] = 1;

  printf("Discrete-action PPO\n");
  F.push_back(new Approximator("policy", S, input, data));
  F.push_back(new Approximator("critic", S, input, data));
  Builder build_pol = F[0]->buildFromSettings(S, aInfo.maxLabel);
  Builder build_val = F[1]->buildFromSettings(S, 1 );

  //build_pol.addParamLayer(1,"Exp",1); //add klDiv penalty coefficient layer

  F[0]->initializeNetwork(build_pol);
  F[1]->initializeNetwork(build_val);
}
