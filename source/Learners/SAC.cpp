//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//
#include "SAC.h"

#include "../Network/Builder.h"
#include "../Utils/StatsTracker.h"
#include "../Math/Continuous_policy.h"
#include "../Network/Approximator.h"
#include "../ReplayMemory/Collector.h"
#include "../Utils/SstreamUtilities.h"

//#define DKL_filter
#define DPG_RETRACE_TGT
#define DPG_LEARN_STDEV

namespace smarties
{

void SAC::Train(const MiniBatch& MB, const Uint wID, const Uint bID) const
{
  const Uint t = MB.sampledTstep(bID), thrID = omp_get_thread_num();

  if(thrID==0) profiler->start("FWD");
  const Rvec pvec = actor->forward(bID, t); // network compute
  const Continuous_policy POL({0, nA+1}, aInfo, pvec);
  const auto & ACT = MB.action(bID, t), & MU = MB.mu(bID, t);
  const Real RHO = POL.importanceWeight(ACT, MU), DKL = POL.KLDivergence(MU);
  const bool isFarPol = isFarPolicy(RHO, CmaxRet, CinvRet);

  critc->setAddedInputType(NETWORK, bID, t, 1); // 1 flags to write on
  const Rvec sval = critc->forward(bID, t, 1); // separate net alloc
  Rvec DPG = isFarPol? Rvec(nA, 0) : critc->oneStepBackProp({1}, bID, t, 1);

  critc->setAddedInputType(ACTION, bID, t);
  const Rvec qval = critc->forward(bID, t); // network compute

  if( MB.isTruncated(bID, t+1) ) {
    const Rvec pNext = actor->forward(bID, t+1);
    critc->setAddedInputType(NETWORK, bID, t+1); // retrace : skip tgt weights
    const Rvec vNext = critc->forward(bID, t+1); // value with state+policy
    MB.updateRetrace(bID, t+1, 0, (vNext[0] + pNext[nA])/2, 0);
  }

  const Real Aest = qval[0] - pvec[nA], Vest = (sval[0] + pvec[nA]) / 2;
  //const Real Vest = (sval[0] + pvec[nA]) / 2, Aest = qval[0] - Vest;
  const Real dQRET = MB.updateRetrace(bID, t, Aest, Vest, RHO);
  const Real Q_RET = MB.Q_RET(bID, t), A_RET = Q_RET - Vest;
  const Real Ver = std::min((Real)1,RHO)*(Q_RET - qval[0] + sval[0] - pvec[nA]);
  const Real Aer = RHO * (Q_RET - qval[0]);

  const Rvec penalG  = POL.KLDivGradient(MU, -1);
  Rvec SPG = isFarPol? Rvec(2*nA, 0) : POL.policyGradient(ACT, A_RET * RHO);
  for (Uint i = 0; i < nA; ++i) {
    SPGnorms[thrID][i] += SPG[i] * SPG[i];
    DPGnorms[thrID][i] += DPG[i] * DPG[i];
    SPG[i] = SPG[i] + 0.5 * DPG[i] * DPGfactor[0];
  }

  Rvec gradPol = Rvec(2*nA + 1, 0);
  gradPol[nA] = isFarPol? 0 : beta * Ver;
  POL.makeNetworkGrad(gradPol, Utilities::weightSum2Grads(SPG, penalG, beta));

  MB.setMseDklImpw(bID, t, Ver*Ver, DKL, RHO, CmaxRet, CinvRet);
  trainInfo->log(qval[0], sval[0] - pvec[nA], SPG, penalG, {dQRET}, thrID);
  actor->setGradient(gradPol, bID, t);
  critc->setGradient({ isFarPol? 0 : Aer }, bID, t);
}

void SAC::select(Agent& agent)
{
  data_get->add_state(agent);
  Sequence& EP = data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(EP);
  for (const auto & net : networks ) net->load(MB, agent, 0);
  const Uint currStep = EP.nsteps() - 1; assert(EP.nsteps()>0);

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    const Rvec pvec = actor->forward(agent);
    const Continuous_policy POL({0, nA+1}, aInfo, pvec);

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    const Rvec action = POL.selectAction(agent, settings.explNoise>0);
    agent.setAction(action);
    data_get->add_action(agent, POL.getVector());

    critc->setAddedInputType(ACTION, agent, currStep);
    const Rvec qval = critc->forward(agent);
    critc->setAddedInputType(NETWORK, agent, currStep);
    const Rvec sval = critc->forward(agent, true); // overwrite = true
    EP.action_adv.push_back( qval[0] - pvec[nA]   );
    EP.state_vals.push_back((sval[0] + pvec[nA])/2);
  }
  else // either terminal or truncation state
  {
    if( agent.agentStatus == TRNC ) { // not a terminal state
      const Rvec pvec = actor->forward(agent); // grab pol mean
      critc->setAddedInputType(NETWORK, agent, currStep);
      const Rvec sval = critc->forward(agent);
      EP.state_vals.push_back((sval[0] + pvec[nA])/2);
    } else {
      EP.state_vals.push_back(0); //value of terminal state is 0
    }
    //whether seq is truncated or terminated, act adv is undefined:
    EP.action_adv.push_back(0);
    const Uint N = EP.nsteps();
    // compute initial Qret for whole trajectory:
    assert(N == EP.action_adv.size() && N == EP.state_vals.size());
    assert(0 == EP.Q_RET.size());
    //within Retrace, we use the Q_RET vector to write the Adv retrace values
    EP.Q_RET.resize(N, 0);
    EP.offPolicImpW.resize(N, 1);
    for(Uint i = EP.ndata(); i > 0; --i)
      EP.propagateRetrace(i, gamma, data->scaledReward(EP, i));

    data_get->terminate_seq(agent);
  }
}

void SAC::setupTasks(TaskQueue& tasks)
{
  if( not bTrain ) return;

  // ALGORITHM DESCRIPTION
  algoSubStepID = -1; // pre initialization

  auto stepInit = [&]()
  {
    // conditions to start the initialization task:
    if ( algoSubStepID >= 0 ) return; // we done with init
    if ( data->readNData() < nObsB4StartTraining ) return; // not enough data to init

    debugL("Initialize Learner");
    initializeLearner();
    algoSubStepID = 0;
    profiler->start("DATA");
  };
  tasks.add(stepInit);

  auto stepMain = [&]()
  {
    // conditions to begin the update-compute task
    if ( algoSubStepID not_eq 0 ) return; // some other op is in progress
    if ( blockGradientUpdates() ) return; // waiting for enough data

    profiler->stop();
    debugL("Sample the replay memory and compute the gradients");
    spawnTrainTasks();
    debugL("Gather gradient estimates from each thread and Learner MPI rank");
    prepareGradient();
    debugL("Search work to do in the Replay Memory");
    processMemoryBuffer(); // find old eps, update avg quantities ...
    debugL("Update Retrace est. for episodes sampled in prev. grad update");
    updateRetraceEstimates();
    debugL("Compute state/rewards stats from the replay memory");
    finalizeMemoryProcessing(); //remove old eps, compute state/rew mean/stdev
    logStats();
    profiler->start("MPI");

    const Real learnRate = settings.learnrate;
    #if 0
    for (Uint i = 0; i < nA; ++i) {
      Real spgNorm = 0, dpgNorm = 0;
      for (Uint j = 0; j < nThreads; ++j) {
        spgNorm += SPGnorms[j][i] / settings.batchSize;
        dpgNorm += DPGnorms[j][i] / settings.batchSize;
      }
      const Real newNorm = std::sqrt(spgNorm / (dpgNorm + nnEPS));
      DPGfactor[i] += settings.learnrate * (newNorm - DPGfactor[i]);
    }
    #else
    Real spgNorm = 0, dpgNorm = 0;
    for (Uint i = 0; i < nA; ++i)
      for (Uint j = 0; j < nThreads; ++j) {
        spgNorm += SPGnorms[j][i] / settings.batchSize / nA;
        dpgNorm += DPGnorms[j][i] / settings.batchSize / nA;
      }
    const Real newNorm = std::sqrt(spgNorm / (dpgNorm + nnEPS));
    DPGfactor[0] += learnRate * (newNorm - DPGfactor[0]);
    #endif

    const Real oldLambda = critc->getOptimizerPtr()->lambda;
    const Real L2critc = critc->getNetworkPtr()->weights->compute_weight_norm();
    const Real L2actor = actor->getNetworkPtr()->weights->compute_weight_norm();
    const Real newLambda = oldLambda * (1 + learnRate * (L2critc/L2actor - 1));
    critc->getOptimizerPtr()->lambda = newLambda;
    //if( ((nGradSteps() + 1) % 1000) == 0 )
    //  printf("%f %f\n", newLambda, DPGfactor[0]);

    algoSubStepID = 1;
  };
  tasks.add(stepMain);

  // these are all the tasks I can do before the optimizer does an allreduce
  auto stepComplete = [&]()
  {
    if ( algoSubStepID not_eq 1 ) return;
    if ( networks[0]->ready2ApplyUpdate() == false ) return;

    profiler->stop();
    debugL("Apply SGD update after reduction of gradients");
    applyGradient();
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
    profiler->start("DATA");
  };
  tasks.add(stepComplete);
}

SAC::SAC(MDPdescriptor& MDP_, Settings& S_, DistributionInfo& D_):
  Learner_approximator(MDP_, S_, D_)
{
  const bool bCreatedEncorder = createEncoder();
  assert(networks.size() == bCreatedEncorder? 1 : 0);
  Approximator* const encoder = bCreatedEncorder? networks[0] : nullptr;
  if(bCreatedEncorder) encoder->initializeNetwork();

  networks.push_back(
    new Approximator("policy", settings, distrib, data.get(), encoder)
  );
  actor = networks.back();
  actor->buildFromSettings(nA + 1);
  const Rvec stdParam = Continuous_policy::initial_Stdev(aInfo, explNoise);
  actor->getBuilder().addParamLayer(nA, "Linear", stdParam);
  actor->initializeNetwork();

  networks.push_back(
    new Approximator("critic", settings, distrib, data.get(), encoder, actor)
  );
  critc = networks.back();
  critc->setAddedInput(NETWORK, nA);
  critc->setNumberOfAddedSamples(1);
  // update settings that are going to be read by critic:
  settings.learnrate *= 10; // DPG wants critic faster than actor
  settings.nnLambda = 1e-4; // also wants L2 penl coef
  settings.nnOutputFunc = "Linear"; // critic must be linear
  critc->buildFromSettings(1);
  critc->initializeNetwork();

  computeQretrace = true;
  trainInfo = new TrainData("SAC", distrib, 1, "| dAdv ", 1);
}

}
