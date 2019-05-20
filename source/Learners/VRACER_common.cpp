//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "../Network/Builder.h"
#include "../Network/Approximator.h"
//#include "../Math/Gaussian_mixture.h"
#include "../Math/Gaussian_policy.h"
#include "../Math/Discrete_policy.h"

namespace smarties
{

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::prepareCMALoss()
{
  if(ESpopSize==1) return;

  profiler->stop_start("LOSS");
  std::vector<Real> aR(batchSize, 0), aA(batchSize, 0);
  #if 1
   #pragma omp parallel for schedule(static)
   for (Uint b=0; b<batchSize; ++b) {
     for(Uint w=0; w<ESpopSize; ++w) { aR[b]+=rhos[b][w]; aA[b]+=advs[b][w]; }
     aR[b] /= ESpopSize; aA[b] /= ESpopSize;
   }
  #else
   for(Uint b=0; b<batchSize; ++b) { aR[b] = rhos[b][0]; aA[b] = advs[b][0]; }
  #endif

  const auto isFar = [&](const Real&W) {return W >= CmaxRet || W <= CinvRet;};
  #pragma omp parallel for schedule(static)
  for (Uint w=0; w<ESpopSize; ++w)
  for (Uint b=0; b<batchSize; ++b) {
    const Real clipR = std::max(CinvRet, std::min(rhos[b][w], CmaxRet));
    const Real clipA = isFar(rhos[b][w]) ? aA[b] : advs[b][w];
    const Real costAdv = - beta * clipR * aA[b]; //minus: to maximize pol adv
    const Real costVal = beta * std::pow(std::min((Real)1, aR[b]) * clipA, 2);
    const Real costDkl = (1-beta) * dkls[b][w];
    networks[0]->ESloss(w) += alpha * (costAdv + costDkl) + (1-alpha) * costVal;
  }
  networks[0]->nAddedGradients = ESpopSize * batchSize;
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::setupNet()
{
  const std::type_info& actT = typeid(Action_t);
  const std::type_info& vecT = typeid(Rvec);
  const bool isContinuous = actT.hash_code() == vecT.hash_code();

  std::vector<Uint> nouts = count_outputs(&aInfo);

  F.push_back(new Approximator("net", settings, input, data));

  #ifdef DACER_simpleSigma // variance not dependent on state
    const Uint varianceSize = nouts.back();
    if(isContinuous) nouts.pop_back();
  #endif

  Builder build = F[0]->buildFromSettings(settings, nouts);

  if(isContinuous)
  {
    Rvec initBias = Rvec(1, 0);
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

template<> std::vector<Uint> VRACER<Discrete_policy, Uint>::
count_outputs(const ActionInfo*const aI) {
  return std::vector<Uint>{1, aI->dimDiscrete()};
}
template<> std::vector<Uint> VRACER<Discrete_policy, Uint>::
count_pol_starts(const ActionInfo*const aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return std::vector<Uint>{indices[1]};
  #else
    return std::vector<Uint>{indices[0]};
  #endif
}
template<> Uint VRACER<Discrete_policy, Uint>::
getnDimPolicy(const ActionInfo*const aI) { return aI->dimDiscrete(); }

template<> VRACER<Discrete_policy, Uint>::
VRACER(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner_approximator(MDP_, S_, D_),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI))
{
  printf("Discrete-action DACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
  computeQretrace = true;
  setupNet();
}

/////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
template<> std::vector<Uint> VRACER<Gaussian_policy, Rvec>::
count_outputs(const ActionInfo*const aI) {
  return std::vector<Uint>{1, aI->dim, aI->dim};
}
template<> std::vector<Uint> VRACER<Gaussian_policy, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return std::vector<Uint>{indices[1], indices[2]};
  #else
    return std::vector<Uint>{indices[0], indices[1]};
  #endif
}
template<> Uint VRACER<Gaussian_policy, Rvec>::
getnDimPolicy(const ActionInfo*const aI) { return 2*aI->dim; }

template<> VRACER<Gaussian_policy, Rvec>::
VRACER(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner_approximator(MDP_, S_, D_),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI)) {
  printf("Gaussian continuous-action V-RACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
  computeQretrace = true;
  setupNet();

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);

    for(Uint i=0; i<mu.size(); ++i) mu[i] = dist(generators[0]);
    for(Uint i=0; i<nA; ++i) mu[i+nA] = std::exp(0.5*mu[i+nA] -1);
    for(Uint i=0; i<nA; ++i) output[1+i] = mu[i] + dist(generators[0])*mu[i+nA];
    for(Uint i=0; i<nA; ++i)
      output[1+i+nA] = noiseMap_inverse(mu[i+nA]) + .1*dist(generators[0]);

    Gaussian_policy pol = prepare_policy<Gaussian_policy>(output);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}


///////////////////////////////////////////////////////////////////////////////
#if 0

template<> std::vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
count_outputs(const ActionInfo*const aI) {
  return std::vector<Uint>{1, NEXPERTS, NEXPERTS*aI->dim, NEXPERTS*aI->dim};
}
template<> std::vector<Uint> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
count_pol_starts(const ActionInfo*const aI) {
  const std::vector<Uint> sizes = count_outputs(aI);
  const std::vector<Uint> indices = count_indices(sizes);
  #ifdef DACER_singleNet
    return std::vector<Uint>{indices[1], indices[2], indices[3]};
  #else
    return std::vector<Uint>{indices[0], indices[1], indices[2]};
  #endif
}
template<> Uint VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
getnDimPolicy(const ActionInfo*const aI) { return NEXPERTS*(1 +2*aI->dim); }

template<> VRACER<Gaussian_mixture<NEXPERTS>, Rvec>::
VRACER(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_): Learner_approximator(MDP_, S_, D_),
net_outputs(count_outputs(&_env->aI)),pol_start(count_pol_starts(&_env->aI))
{
  printf("Mixture-of-experts continuous-action V-RACER: Built network with outputs: v:%u pol:%s\n", VsID, print(pol_start).c_str());
  computeQretrace = true;
  setupNet();

  {  // TEST FINITE DIFFERENCES:
    Rvec output(F[0]->nOutputs()), mu(getnDimPolicy(&aInfo));
    std::normal_distribution<Real> dist(0, 1);
    for(Uint i=0; i<output.size(); ++i) output[i] = dist(generators[0]);
    for(Uint i=0; i<mu.size(); ++i) mu[i] = dist(generators[0]);
    Real norm = 0;
    for(Uint i=0; i<NEXPERTS; ++i) {
      mu[i] = std::exp(mu[i]);
      norm += mu[i];
    }
    for(Uint i=0; i<NEXPERTS; ++i) mu[i] = mu[i]/norm;
    for(Uint i=NEXPERTS*(1+nA);i<NEXPERTS*(1+2*nA);++i) mu[i]=std::exp(mu[i]);

    Gaussian_mixture<NEXPERTS> pol = prepare_policy<Gaussian_mixture<NEXPERTS>>(output);
    Rvec act = pol.finalize(1, &generators[0], mu);
    pol.prepare(act, mu);
    pol.test(act, mu);
  }
}
#endif

///////////////////////////////////////////////////////////////////////////////

}
