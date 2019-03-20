//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "VRACER.h"

#include "VRACER_common.cpp"
#include "VRACER_train.cpp"

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::select(Agent& agent)
{
  Sequence* const S = data_get->get(agent.ID);
  data_get->add_state(agent);
  F[0]->prepare_agent(S, agent);

  #ifdef DACER_singleNet
    static constexpr int valNetID = 0;
  #else
    static constexpr int valNetID = 1;
    F[1]->prepare_agent(S, agent);
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
    Policy_t pol = prepare_policy<Policy_t>(output);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    auto act = pol.finalize(explNoise>0, &generators[nThreads+agent.ID], mu);

    S->state_vals.push_back(value[0]);
    agent.act(act);
    data_get->add_action(agent, mu);
  }
  else
  {
    if( agent.Status == TRNC_COMM ) {
      Rvec output = F[valNetID]->forward_agent(agent);
      S->state_vals.push_back(output[0]);
    } else S->state_vals.push_back(0); //value of term state is 0

    const Uint N = S->nsteps();
    // compute initial Qret for whole trajectory:
    assert(N==S->state_vals.size());
    assert(0==S->Q_RET.size() && 0==S->action_adv.size());

    // compute initial Qret for whole trajectory
    //within Retrace, we use the state_vals vector to write the Q retrace values
    //both if truncated or not, last delta is zero
    S->Q_RET.resize(N, 0);
    S->action_adv.resize(N, 0);
    S->offPolicImpW.resize(N, 1);
    for(Uint i=S->ndata(); i>0; i--) backPropRetrace(S, i);

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
}

template<typename Policy_t, typename Action_t>
void VRACER<Policy_t, Action_t>::setupTasks(TaskQueue& tasks)
{
  if( not bTrain ) return;

  // ALGORITHM DESCRIPTION
  algoSubStepID = -1; // pre initialization
  {
    auto condInit = [&]() {
      if( algoSubStepID>=0 ) return false;
      return data->readNData() >= nObsB4StartTraining;
    };
    auto stepInit = [&]() {
      debugL("Initialize Learner");
      initializeLearner();
      algoSubStepID = 0;
    };
    tasks.add(condInit, stepInit);
  }

  {
    auto condMain = [&]() {
      if( algoSubStepID != 0 ) return false;
      else return unblockGradientUpdates();
    };
    auto stepMain = [&]() {
      debugL("Sample the replay memory and compute the gradients");
      spawnTrainTasks();
      prepareCMALoss();
      debugL("Gather gradient estimates from each thread and Learner MPI rank");
      prepareGradient();
      debugL("Search work to do in the Replay Memory");
      processMemoryBuffer(); // find old eps, update avg quantities ...
      debugL("Update Retrace est. for episodes sampled in prev. grad update");
      updateRetraceEstimates();
      debugL("Compute state/rewards stats from the replay memory");
      finalizeMemoryProcessing(); //remove old eps, compute state/rew mean/stdev
      logStats();
      algoSubStepID = 1;
    };
    tasks.add(condMain, stepMain);
  }
  // these are all the tasks I can do before the optimizer does an allreduce
  {
    auto condComplete = [&]() {
      if(algoSubStepID != 1) return false;
      // assumption here is that I have at least one approximator
      // and it that one is ready to apply update they are all/will be soon
      return F[0]->ready2ApplyUpdate();
    };
    auto stepComplete = [&]() {
      debugL("Apply SGD update after reduction of gradients");
      applyGradient();
      algoSubStepID = 0; // rinse and repeat
      globalGradCounterUpdate(); // step ++
    };
    tasks.add(condComplete, stepComplete);
  }
}

///////////////////////////////////////////////////////////////////////////////

template class VRACER<Discrete_policy, Uint>;
template class VRACER<Gaussian_mixture<NEXPERTS>, Rvec>;
template class VRACER<Gaussian_policy, Rvec>;
