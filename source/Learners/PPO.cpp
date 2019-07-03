//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "PPO.h"

template class PPO<Discrete_policy, Uint>;
template class PPO<Gaussian_policy, Rvec>;
using PPO_contAct = PPO<Gaussian_policy, Rvec>;
using PPO_discAct = PPO<Discrete_policy, Uint>;

#include "PPO_common.cpp"
#include "PPO_train.cpp"

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::select(Agent& agent)
{
  data_get->add_state(agent);
  Sequence& EP = * data_get->get(agent.ID);
  const MiniBatch MB = data->agentToMinibatch(&EP);

  if( agent.agentStatus < TERM ) // not end of sequence
  {
    actor->load(MB, agent, 0);
    critc->load(MB, agent, 0);
    //Compute policy and value on most recent element of the sequence.
    Gaussian_policy POL = prepare_policy(actor->forward(agent));
    const Rvec sval = critc->forward_agent(agent);
    EP.state_vals.push_back(sval[0]); // not a terminal state
    Rvec MU = POL.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    const bool bSamplePolicy = settings.explNoise > 0;
    auto act = POL.finalize(bSamplePolicy, &generators[nThreads+agent.ID], MU);
    agent.act(act);
    data_get->add_action(agent, MU);
  }
  else if( agent.agentStatus == TRNC )
  {
    critc->load(MB, agent, 0);
    const Rvec sval = critc->forward_agent(agent);
    EP.state_vals.push_back(sval[0]); // not a terminal state
  }
  else // TERM state
    EP.state_vals.push_back(0); //value of terminal state is 0

  updatePPO(EP);

  //advance counters of available data for training
  if(agent.agentStatus >= TERM) data_get->terminate_seq(agent);
}

template<typename Policy_t, typename Action_t>
bool PPO<Policy_t, Action_t>::blockDataAcquisition() const
{
  return data->readNData() >= nHorizon + cntKept;
}

template<typename Policy_t, typename Action_t>
void PPO<Policy_t, Action_t>::setupTasks(TaskQueue& tasks)
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
    initializeGAE(); // rescale GAE with learner rewards scale
    algoSubStepID = 0;
  };
  tasks.add(stepInit);

  auto stepMain = [&]()
  {
    // conditions to begin the update-compute task
    if ( algoSubStepID not_eq 0 ) return; // some other op is in progress
    if ( blockGradientUpdates() ) return; // waiting for enough data

    debugL("Sample the replay memory and compute the gradients");
    spawnTrainTasks();
    debugL("Gather gradient estimates from each thread and Learner MPI rank");
    prepareGradient();
    updatePenalizationCoef();
    advanceEpochCounters();
    debugL("Search work to do in the Replay Memory");
    processMemoryBuffer(); // find old eps, update avg quantities ...
    debugL("Update Retrace est. for episodes sampled in prev. grad update");
    updateRetraceEstimates();
    debugL("Compute state/rewards stats from the replay memory");
    finalizeMemoryProcessing(); //remove old eps, compute state/rew mean/stdev
    logStats();
    algoSubStepID = 1;
  };
  tasks.add(stepMain);

  // these are all the tasks I can do before the optimizer does an allreduce
  auto stepComplete = [&]()
  {
    if ( algoSubStepID not_eq 1 ) return;
    if ( networks[0]->ready2ApplyUpdate() == false ) return;

    debugL("Apply SGD update after reduction of gradients");
    applyGradient();
    algoSubStepID = 0; // rinse and repeat
    globalGradCounterUpdate(); // step ++
  };
  tasks.add(stepComplete);
}
