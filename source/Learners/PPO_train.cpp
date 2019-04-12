//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::prepareGradient()
{
  if(not updateComplete || updateToApply) die("undefined behavior");
  if(learn_rank > 0)
    die("This method does not support multiple learner ranks yet");

  Learner::prepareGradient();

  debugL("update lagrangian penalization coefficient");
  cntPenal[0] = 0;
  for(Uint i=1; i<=nThreads; i++) {
    cntPenal[0] += cntPenal[i]; cntPenal[i] = 0;
  }
  if(cntPenal[0]<nnEPS) die("undefined behavior");
  const Real fac = learnR/cntPenal[0]; // learnRate*grad/N //
  cntPenal[0] = 0;
  for(Uint i=1; i<=nThreads; i++) {
      valPenal[0] += fac*valPenal[i];
      valPenal[i] = 0;
  }
  if(valPenal[0] <= nnEPS) valPenal[0] = nnEPS;


  const Uint currStep = nStep()+1; //base class will advance this with this func
  debugL("shift counters of epochs over the stored data");
  cntBatch += batchSize;
  if(cntBatch >= nHorizon) {
    const Real annlLR = annealRate(learnR, currStep, epsAnneal);
    data_proc->updateRewardsStats(0.001, annlLR);
    cntBatch = 0;
    cntEpoch++;
  }

  if(cntEpoch >= nEpochs) {
    debugL("finished epochs, compute state/rew stats, clear buffer to gather new onpol samples");
    #if 0 // keep nearly on policy data
      cntKept = data->clearOffPol(CmaxPol, 0.05);
    #else
      data->clearAll();
      cntKept = 0;
    #endif
    //reset batch learning counters
    cntEpoch = 0;
    cntBatch = 0;
  }
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::Train(const Uint seq, const Uint samp,
  const Uint wID, const Uint bID, const Uint thrID) const
{
  if(thrID==0)  profiler->stop_start("FWD");
  Sequence* const traj = data->get(seq);
  const Real adv_est = traj->action_adv[samp], val_tgt = traj->Q_RET[samp];
  const Rvec MU = traj->tuples[samp]->mu;

  F[0]->prepare_one(traj, samp, thrID, wID);
  const Rvec pol_cur = F[0]->forward(samp, thrID);

  if(thrID==0)  profiler->stop_start("CMP");

  const Policy_t pol = prepare_policy<Policy_t>(pol_cur, &aInfo, pol_indices, traj->tuples[samp]);
  const Real rho_cur = pol.sampImpWeight, DivKL = pol.sampKLdiv;
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

  F[1]->prepare_one(traj, samp, thrID, wID);
  const Rvec val_cur = F[1]->forward(samp, thrID);

  #ifdef PPO_PENALKL //*nonZero(gain)
    const Rvec policy_grad = pol.policy_grad(pol.sampAct, gain);
    const Rvec penal_grad = pol.div_kl_grad(MU, -valPenal[0]);
    const Rvec totalPolGrad = sum2Grads(penal_grad, policy_grad);
  #else //we still learn the penal coef, for simplicity, but no effect
    const Rvec totalPolGrad = pol.policy_grad(pol.sampAct, gain);
    const Rvec policy_grad = totalPolGrad;
    const Rvec penal_grad = Rvec(policy_grad.size(), 0);
  #endif

  assert(wID == 0);
  Rvec grad(F[0]->nOutputs(), 0);
  pol.finalize_grad(totalPolGrad, grad);

  //bookkeeping:
  const Real verr = val_tgt-val_cur[0];
  #ifdef PPO_learnDKLt
  trainInfo->log(val_cur[0], verr, policy_grad, penal_grad,
    { (Real)valPenal[0], DivKL, rho_cur, DKL_target }, thrID);
  #else
  trainInfo->log(val_cur[0], verr, policy_grad, penal_grad,
    { (Real)valPenal[0], DivKL, rho_cur }, thrID);
  #endif
  traj->setMseDklImpw(samp, verr*verr, DivKL, rho_cur, 1+CmaxPol, 1-CmaxPol);

  if(thrID==0)  profiler->stop_start("BCK");
  //if(!thrID) cout << "back pol" << endl;
  F[0]->backward(grad, samp, thrID);
  //if(!thrID) cout << "back val" << endl; //*(!isFarPol)
  F[1]->backward({verr*(!isFarPol)}, samp, thrID);
  F[0]->gradient(thrID);
  F[1]->gradient(thrID);
}

template<>
vector<Uint> PPO<Discrete_policy, Uint>::count_pol_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{aI->maxLabel};
}
template<>
vector<Uint> PPO<Discrete_policy, Uint>::count_pol_starts(const ActionInfo*const aI)
{
  const vector<Uint> indices = count_indices(count_pol_outputs(aI));
  return vector<Uint>{indices[0]};
}
template<>
Uint PPO<Discrete_policy, Uint>::getnDimPolicy(const ActionInfo*const aI)
{
  return aI->maxLabel;
}

template<>
vector<Uint> PPO<Gaussian_policy, Rvec>::count_pol_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{aI->dim, aI->dim};
}
template<>
vector<Uint> PPO<Gaussian_policy, Rvec>::count_pol_starts(const ActionInfo*const aI)
{
  const vector<Uint> indices = count_indices(count_pol_outputs(aI));
  return vector<Uint>{indices[0], indices[1]};
}
template<>
Uint PPO<Gaussian_policy, Rvec>::getnDimPolicy(const ActionInfo*const aI)
{
  return 2*aI->dim;
}

template<> PPO<Gaussian_policy, Rvec>::PPO(Environment*const E, Settings&S) :
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

template<> PPO<Discrete_policy, Uint>::PPO(Environment*const E, Settings&S) :
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
