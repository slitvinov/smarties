//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#include "../Network/Builder.h"
#include "../Network/Approximator.h"

#ifdef ADV_GAUS
#include "../Math/Mixture_advantage_gaus.h"
#include "../Math/Gaus_advantage.h"
#else
#include "../Math/Mixture_advantage_quad.h"
#include "../Math/Quadratic_advantage.h"
#endif
#include "../Math/Discrete_advantage.h"

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::setupNet()
{
  const std::type_info& actT = typeid(Action_t);
  const std::type_info& vecT = typeid(Rvec);
  const bool isContinuous = actT.hash_code() == vecT.hash_code();

  vector<Uint> nouts = count_outputs(&aInfo);

  #ifdef RACER_singleNet // state value is approximated by an other net
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

  #ifdef RACER_simpleSigma // variance not dependent on state
    const Uint varianceSize = nouts.back();
    if(isContinuous) nouts.pop_back();
  #endif

  Builder build = F[0]->buildFromSettings(settings, nouts);

  if(isContinuous)
  {
    #ifdef RACER_singleNet
      Rvec  polBias = Rvec(1, 0);
      Rvec& valBias = polBias;
    #else
      Rvec polBias = Rvec(0, 0); // no state val here
      Rvec valBias = Rvec(1, 0); // separate bias init vector for val net
    #endif
    Advantage_t::setInitial(&aInfo, valBias);
    Policy_t::setInitial_noStdev(&aInfo, polBias);

    #ifdef RACER_simpleSigma // sigma not linked to state: param output
      build.setLastLayersBias(polBias);
      #ifdef EXTRACT_COVAR
        Real initParam = noiseMap_inverse(explNoise*explNoise);
      #else
        Real initParam = noiseMap_inverse(explNoise);
      #endif
      build.addParamLayer(varianceSize, "Linear", initParam);
    #else
      Policy_t::setInitial_Stdev(&aInfo, polBias, explNoise);
      build.setLastLayersBias(polBias);
    #endif
  }

  // construct policy net:
  if(F.size() > 1) die("");
  F[0]->initializeNetwork(build);
  F[0]->opt->bAnnealLearnRate= true;
  trainInfo = new TrainData("racer", settings, 1, "| dAdv | avgW ", 2);
}

// Template specializations. From now on, nothing relevant to algorithm itself.

template<> vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_outputs(const ActionInfo*const aI) {
  return vector<Uint>{1, aI->maxLabel, aI->maxLabel};
}
template<> vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[2]};
}
template<> vector<Uint>
RACER<Discrete_advantage, Discrete_policy, Uint>::
count_adv_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1]};
}
template<> Uint
RACER<Discrete_advantage, Discrete_policy, Uint>::
getnOutputs(const ActionInfo*const aI) {
  return 1 + aI->maxLabel + aI->maxLabel;
}
template<> Uint
RACER<Discrete_advantage, Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo*const aI) {
  return aI->maxLabel;
}

template<>
RACER<Discrete_advantage, Discrete_policy, Uint>::
RACER(Environment*const _env, Settings& _set) : Learner_approximator(_env,_set),
  net_outputs(count_outputs(&_env->aI)),
  pol_start(count_pol_starts(&_env->aI)),
  adv_start(count_adv_starts(&_env->aI))
{
  if(_set.learner_rank == 0) {
    printf("Discrete-action RACER: Built network with outputs: v:%u pol:%s adv:%s\n", VsID, print(pol_start).c_str(), print(adv_start).c_str());
  }
  computeQretrace = true;
  setupNet();
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

template<> vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_outputs(const ActionInfo*const aI) {
  const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(aI);
  return vector<Uint>{1, nL, NEXPERTS, NEXPERTS*aI->dim, NEXPERTS*aI->dim};
}
template<> vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[2], indices[3], indices[4]};
}
template<> vector<Uint>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
count_adv_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1]};
}
template<> Uint
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
getnOutputs(const ActionInfo*const aI) {
  const Uint nL = Mixture_advantage<NEXPERTS>::compute_nL(aI);
  return 1 + nL + NEXPERTS*(1 +2*aI->dim);
}
template<> Uint
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
getnDimPolicy(const ActionInfo*const aI) {
  return NEXPERTS*(1 +2*aI->dim);
}

template<>
RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>::
RACER(Environment*const _env, Settings& _set) : Learner_approximator(_env,_set),
  net_outputs(count_outputs(&_env->aI)),
  pol_start(count_pol_starts(&_env->aI)),
  adv_start(count_adv_starts(&_env->aI))
{
  if(_set.learner_rank == 0) {
    printf("Mixture-of-experts continuous-action RACER: Built network with outputs: v:%u pol:%s adv:%s (sorted %s)\n", VsID, print(pol_start).c_str(), print(adv_start).c_str(), print(net_outputs).c_str());
  }
  computeQretrace = true;
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

    auto pol = prepare_policy<Gaussian_mixture<NEXPERTS>>(output);
    Rvec act = pol.finalize(1, &generators[0], mu);
    auto adv = prepare_advantage<Mixture_advantage<NEXPERTS>>(output, &pol);
    adv.test(act, &generators[0]);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////

template<> vector<Uint>
RACER<Param_advantage, Gaussian_policy, Rvec>::
count_outputs(const ActionInfo*const aI) {
  const Uint nL = Param_advantage::compute_nL(aI);
  return vector<Uint>{1, nL, aI->dim, aI->dim};
}
template<> vector<Uint>
RACER<Param_advantage, Gaussian_policy, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[2], indices[3]};
}
template<> vector<Uint>
RACER<Param_advantage, Gaussian_policy, Rvec>::
count_adv_starts(const ActionInfo*const aI) {
  const vector<Uint> sizes = count_outputs(aI);
  const vector<Uint> indices = count_indices(sizes);
  return vector<Uint>{indices[1]};
}
template<> Uint
RACER<Param_advantage, Gaussian_policy, Rvec>::
getnOutputs(const ActionInfo*const aI) {
  const Uint nL = Param_advantage::compute_nL(aI);
  return 1 + nL + 2*aI->dim;
}
template<> Uint
RACER<Param_advantage, Gaussian_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI) {
  return 2*aI->dim;
}

template<>
RACER<Param_advantage, Gaussian_policy, Rvec>::
RACER(Environment*const _env, Settings& _set) : Learner_approximator(_env,_set),
  net_outputs(count_outputs(&_env->aI)),
  pol_start(count_pol_starts(&_env->aI)),
  adv_start(count_adv_starts(&_env->aI))
{
  if(_set.learner_rank == 0) {
    printf("Gaussian continuous-action RACER: Built network with outputs: v:%u pol:%s adv:%s (sorted %s)\n", VsID, print(pol_start).c_str(), print(adv_start).c_str(), print(net_outputs).c_str());
  }
  computeQretrace = true;
  setupNet();

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);

    for(Uint i=0; i<mu.size(); i++) mu[i] = dist(generators[0]);
    for(Uint i=0; i<nA; i++) mu[i+nA] = std::exp(0.5*mu[i+nA] -1);

    for(Uint i=0; i<=nL; i++) output[i] = 0.5*dist(generators[0]);
    for(Uint i=0; i<nA; i++)
      output[1+nL+i] = mu[i] + dist(generators[0])*mu[i+nA];
    for(Uint i=0; i<nA; i++)
      output[1+nL+i+nA] = noiseMap_inverse(mu[i+nA]) + .1*dist(generators[0]);

    auto pol = prepare_policy<Gaussian_policy>(output);
    Rvec act = pol.finalize(1, &generators[0], mu);
    auto adv = prepare_advantage<Param_advantage>( output, &pol );
    adv.test(act, &generators[0]);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}
