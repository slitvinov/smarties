//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "CMALearner.h"
#include "../Network/Builder.h"

// nHorizon is number of obs in buffer during training
// steps per epoch = nEpochs * nHorizon / batchSize
// obs per step = nHorizon / (steps per epoch)
// this leads to formula used to compute nEpochs
template<typename Action_t>
void CMALearner<Action_t>::select(Agent& agent)
{
  Sequence*const curr_seq = data_get->get(agent.ID);
  data_get->add_state(agent);

  const Uint wrkr = agent.workerID;

  if(agent.Status == INIT_COMM and WnEnded[wrkr] != nAgentsPerWorker) die("");
  if(agent.Status == CONT_COMM and WnEnded[wrkr]>0) WnEnded[wrkr] = 0;

  const Uint progress = WiEnded[wrkr] * nWorkers_own + wrkr;
  const Uint weightID = ESpopStart + progress / batchSize;
  //printf("Using weight %u on worker %u\n", weightID, wrkr);

  if( agent.Status <  TERM_COMM ) //non terminal state
  {
    //Compute policy and value on most recent element of the sequence:
    F[0]->prepare_agent(curr_seq, agent, weightID);
    const Rvec pol = F[0]->forward_agent(agent);
    agent.act(pol);
    data_get->add_action(agent, pol);
  }
  else
  {
    R[wrkr][weightID] += agent.cumulative_rewards;
    data_get->terminate_seq(agent);
    //_warn("%u %u %f",wrkr, weightID, R[wrkr][weightID]);
    ++WnEnded[wrkr];
    if(WnEnded[wrkr] > nAgentsPerWorker) die("");

    if(WnEnded[wrkr] == nAgentsPerWorker) {
      //printf("Simulation %ld ended for worker %u\n", WnEnded[wrkr], wrkr);
      ++WiEnded[wrkr];
    }

    if(WiEnded[wrkr] >= nSeqPerWorker) {
      const auto myStep = _nStep.load();
      while(myStep == _nStep.load()) usleep(5);
      //_warn("worker %u done wait", wrkr);
      WiEnded[wrkr] = 0;
    }
  }
}

template<typename Action_t>
void CMALearner<Action_t>::prepareGradient()
{
  updateComplete = true;
  //fflush(stdout); fflush(0);
  #pragma omp parallel for schedule(static)
  for (Uint w=0; w<ESpopSize; w++) {
    for (Uint b=0; b<nWorkers_own; b++) F[0]->losses[w] -= R[b][w];
    //std::cout << F[0]->losses[w] << std::endl;
  }

  R = std::vector<Rvec>(nWorkers_own, Rvec(ESpopSize, 0) );
  F[0]->nAddedGradients = nWorkers_own * ESpopSize;

  Learner::prepareGradient();

  debugL("shift counters of epochs over the stored data");
  profiler->stop_start("PRE");
  data_proc->updateRewardsStats(0.001, 0.001);
}
template<typename Action_t>
void CMALearner<Action_t>::applyGradient()
{
  Learner::applyGradient();
  //warn("clearing that data");
  data->clearAll();
  _nStep++;
  bUpdateNdata = false;
}
template<typename Action_t>
void CMALearner<Action_t>::globalGradCounterUpdate() {}

template<typename Action_t>
void CMALearner<Action_t>::initializeLearner()
{
  //nothing to do
}

template<typename Action_t>
bool CMALearner<Action_t>::blockDataAcquisition() const
{
  return data->readNSeq() >= nSeqPerStep;
}

template<typename Action_t>
void CMALearner<Action_t>::spawnTrainTasks_seq()
{
}

template<typename Action_t>
void CMALearner<Action_t>::spawnTrainTasks_par()
{
}

template<typename Action_t>
bool CMALearner<Action_t>::bNeedSequentialTrain() { return true; }

template<>
vector<Uint> CMALearner<Uint>::count_pol_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{aI->maxLabel};
}
template<>
vector<Uint> CMALearner<Uint>::count_pol_starts(const ActionInfo*const aI)
{
  return vector<Uint>{0};
}
template<>
Uint CMALearner<Uint>::getnDimPolicy(const ActionInfo*const aI)
{
  return aI->maxLabel;
}

template<>
vector<Uint> CMALearner<Rvec>::count_pol_outputs(const ActionInfo*const aI)
{
  return vector<Uint>{aI->dim};
}
template<>
vector<Uint> CMALearner<Rvec>::count_pol_starts(const ActionInfo*const aI)
{
  return vector<Uint>{0};
}
template<>
Uint CMALearner<Rvec>::getnDimPolicy(const ActionInfo*const aI)
{
  return aI->dim;
}

template<> CMALearner<Rvec>::CMALearner(Environment*const E, Settings& S) :
Learner(E, S)
{
  data_get = new Collector(S, this, data);
  bReady4Init = true;
  tPrint = 10;
  printf("CMALearner\n");
  F.push_back(new Approximator("policy", S, input, data));
  Builder build_pol = F[0]->buildFromSettings(S, {aInfo.dim});
  F[0]->initializeNetwork(build_pol);
}

template<> CMALearner<Uint>::CMALearner(Environment*const E, Settings & S) :
Learner(E, S)
{
  data_get = new Collector(S, this, data);
  bReady4Init = true;
  tPrint = 10;
  printf("Discrete-action CMALearner\n");
  F.push_back(new Approximator("policy", S, input, data));
  Builder build_pol = F[0]->buildFromSettings(S, aInfo.maxLabel);
  F[0]->initializeNetwork(build_pol);
}

template class CMALearner<Uint>;
template class CMALearner<Rvec>;
