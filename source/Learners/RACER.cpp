//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "RACER.h"
#include "RACER_common.cpp"
#include "RACER_train.cpp"
#include "Utils/FunctionUtilities.h"

namespace smarties
{

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::
select(Agent& agent)
{
  Sequence* const traj = data_get->get(agent.ID);
  data_get->add_state(agent);
  F[0]->prepare_agent(traj, agent);

  if( agent.Status < TERM_COMM ) // not end of sequence
  {
    //Compute policy and value on most recent element of the sequence.
    Rvec output = F[0]->forward_agent(agent);
    auto pol = prepare_policy<Policy_t>(output);
    Rvec mu = pol.getVector(); // vector-form current policy for storage

    // if explNoise is 0, we just act according to policy
    // since explNoise is initial value of diagonal std vectors
    // this should only be used for evaluating a learned policy
    Action_t act = pol.finalize(explNoise>0, &generators[nThreads+agent.ID],mu);
    const auto adv = prepare_advantage<Advantage_t>(output, &pol);
    const Real advantage = adv.computeAdvantage(pol.sampAct);
    traj->action_adv.push_back(advantage);
    traj->state_vals.push_back(output[VsID]);
    agent.act(act);
    data_get->add_action(agent, mu);

    #ifndef NDEBUG
      auto dbg = prepare_policy<Policy_t>(output);
      const Rvec & ACT = traj->actions.back(), & MU = traj->policies.back();
      dbg.prepare(ACT, MU);
      const double err = fabs(dbg.sampImpWeight-1);
      if(err>1e-10) _die("Imp W err %20.20e", err);
    #endif
  }
  else // either terminal or truncation state
  {
    if( agent.Status == TRNC_COMM ) {
      Rvec output = F[0]->forward_agent(agent);
      traj->state_vals.push_back(output[VsID]); // not a terminal state
    } else {
      traj->state_vals.push_back(0); //value of terminal state is 0
    }
    //whether seq is truncated or terminated, act adv is undefined:
    traj->action_adv.push_back(0);
    const Uint N = traj->nsteps();
    // compute initial Qret for whole trajectory:
    assert(N == traj->action_adv.size());
    assert(N == traj->state_vals.size());
    assert(0 == traj->Q_RET.size());
    //within Retrace, we use the Q_RET vector to write the Adv retrace values
    traj->Q_RET.resize(N, 0); traj->offPolicImpW.resize(N, 1);
    for(Uint i=traj->ndata(); i>0; i--) backPropRetrace(traj, i);

    OrUhState[agent.ID] = Rvec(nA, 0); //reset temp. corr. noise
    data_get->terminate_seq(agent);
  }
}

template<typename Advantage_t, typename Policy_t, typename Action_t>
void RACER<Advantage_t, Policy_t, Action_t>::setupTasks(TaskQueue& tasks)
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

////////////////////////////////////////////////////////////////////////////

template class RACER<Discrete_advantage, Discrete_policy, Uint>;
template class RACER<Param_advantage, Gaussian_policy, Rvec>;

//template class RACER<Mixture_advantage<NEXPERTS>, Gaussian_mixture<NEXPERTS>, Rvec>;

}
