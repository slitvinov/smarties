//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the MIT license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Scheduler.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

Master::Master(Communicator_internal* const _c, const std::vector<Learner*> _l,
  Environment*const _e, Settings&_s): settings(_s),comm(_c),learners(_l),env(_e)
{
  if(nWorkers_own*nPerRank != static_cast<int>(agents.size()))
    die("Mismatch in master's nWorkers nPerRank nAgents.");

  worker_replies.reserve(nWorkers_own);
  //the following Irecv will be sent after sending the action
  for(int i=1; i<=nWorkers_own; i++) comm->recvBuffer(i);

  for(const auto& L : learners) L->setupTasks(tasks);
}

void Master::run()
{
  {
    #pragma omp parallel num_threads(nThreads)
    {
      std::vector<int> shareWorkers;
      const int thrID = omp_get_thread_num(), thrN = omp_get_num_threads();
      for(int i=1; i<=nWorkers_own; i++)
       if( thrID == (( ( (-i)%thrN ) +thrN ) %thrN) ) shareWorkers.push_back(i);

      #pragma omp critical
      if(shareWorkers.size()) worker_replies.push_back(
        std::thread( [&, shareWorkers] () { processWorker(shareWorkers); }));
    }
  }

  Uint minNdataB4Train = learners[0]->nObsB4StartTraining;
  int firstLearnerStart = 0, isStarted = 0, percentageReady = -5;
  for(size_t i=1; i<learners.size(); i++) {
    Uint tmp = learners[i]->nObsB4StartTraining;
    if(tmp<minNdataB4Train) {
      minNdataB4Train = tmp;
      firstLearnerStart = i;
    }
  }
  const auto isTrainingStarted = [&]() {
    if(not isStarted && learn_rank==0) {
      const auto nCollected = learners[firstLearnerStart]->locDataSetSize();
      const int currPerc = nCollected * 100./(Real) minNdataB4Train;
      if(nCollected >= minNdataB4Train) isStarted = true;
      else if(currPerc > percentageReady+5) {
       percentageReady = currPerc;
       printf("\rCollected %d%% of data required to begin training. ",currPerc);
       fflush(0); //otherwise no show on some platforms
      }
    }
  };
  const auto isTrainingOver = [&](const Learner* const L) {
    // if agents share learning algo, return number of turns performed by env
    // instead of sum of timesteps performed by each agent
    const Real factor = learners.size() == 1? 1.0/nPerRank : 1;
    const long dataCounter = bTrain? L->nLocTimeStepsTrain() : L->nSeqsEval();
    return dataCounter * factor >= totNumSteps;
  };

  while(1)
  {
    isTrainingStarted();

    tasks.run();

    bool over = true;
    for(const auto& L : learners) over = over && isTrainingOver(L);
    if (over) break;
  }
}

void Master::processWorker(const std::vector<int> workers)
{
  while(1)
  {
    for( const int worker : workers )
    {
      assert(worker>0 && worker <= (int) nWorkers_own);
      int completed = comm->testBuffer(worker);

      // Learners lock workers queue if they have enough data to advance step
      while ( bTrain && completed && learnersLockQueue() ) {
        usleep(1);
        if( bExit.load() > 0 ) break;
      }

      if(completed) processAgent(worker);

      usleep(1);
    }
  }
}

void Master::processAgent(const int worker)
{
  //read from worker's buffer:
  std::vector<double> recv_state(sI.dim);
  int recv_agent  = -1; // id of agent inside environment
  int recv_status = -1; // initial/normal/termination/truncation of episode
  double reward   =  0;
  comm->unpackState(worker-1, recv_agent, recv_status, recv_state, reward);
  if (recv_status == FAIL_COMM) die("app crashed");

  const int agent = (worker-1) * nPerRank + recv_agent;
  Learner*const learner = pickLearner(agent, recv_agent);

  agents[agent]->update(recv_status, recv_state, reward);
  //pick next action and ...do a bunch of other stuff with the data:
  learner->select(*agents[agent]);

  debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]",
    agent, agents[agent]->Status, agents[agent]->sOld._print().c_str(),
    agents[agent]->s._print().c_str(), agents[agent]->r,
    agents[agent]->a._print().c_str());

  std::vector<double> actVec = agents[agent]->getAct();
  if(agents[agent]->Status >= TERM_COMM) {
    const Real factor = learners.size() == 1? 1.0/nPerRank : 1;
    const auto nSteps = learner->nLocTimeStepsTrain();
    actVec[0] = factor * nSteps;
    dumpCumulativeReward(recv_agent, worker, learner->nGradSteps(), nSteps);
  }

  debugS("Sent action to worker %d: [%s]", worker, print(actVec).c_str() );
  comm->sendBuffer(worker, actVec);
  comm->recvBuffer(worker); // prepare next recv
}

bool Master::learnersLockQueue() const
{
  //When would a learning algo stop acquiring more data?
  //Off Policy algos:
  // - User specifies a ratio of observed trajectories to gradient steps.
  //    Comm is restarted or paused to maintain this ratio consant.
  //On Policy algos:
  // - if collected enough trajectories for current batch, then comm is paused
  //    untill gradient is applied (or nepocs are done), then comm restarts
  //    to obtain fresh on policy samples
  // Note:
  // - on policy traj. storage assumes that when agent reaches terminal state
  //    on a worker, all other agents on that worker must send their term state
  //    before sending any new initial state

  // However, no learner can stop others from getting data (vector of algos)
  bool locked = true;
  for(const auto& L : learners)
    locked = locked && L->blockDataAcquisition(); // if any wants to unlock...

  return locked;
}

Worker::Worker(Communicator_internal*const _c,Environment*const _e,Settings&_s)
: comm(_c), env(_e), bTrain(_s.bTrain), status(_e->agents.size(),1) {}

void Worker::run()
{
  while(true) {

    while(true) {
      if (comm->recvStateFromApp()) break; //sim crashed

      if (comm->sendActionToApp() ) {
        die("Worker exiting");
        return;
      }
    }
    die("Simulation crash");
    //if we are training, then launch again, otherwise exit
    //if (!bTrain) return;
    comm->launch();
  }
}

void Master::dumpCumulativeReward(const int agent, const int worker,
  const unsigned giter, const unsigned tstep) const
{
  if (giter == 0 && bTrain) return;

  const int ID = (worker-1) * nPerRank + agent;
  char path[256];
  sprintf(path, "agent_%02d_rank%02d_cumulative_rewards.dat", agent,learn_rank);

  std::lock_guard<std::mutex> lock(dump_mutex);
  FILE * pFile = fopen (path, "a");
  fprintf (pFile, "%u %u %d %d %f\n", giter, tstep, worker,
    agents[ID]->transitionID, agents[ID]->cumulative_rewards);
  fflush (pFile);
  fclose (pFile);
}

/*
Client::Client(Learner*const _l, Communicator*const _c, Environment*const _e,
    Settings& _s):
    learner(_l), comm(_c), env(_e), agents(_e->agents), aI(_e->aI), sI(_e->sI),
    sOld(_e->sI), sNew(_e->sI), aOld(_e->aI, &_s.generators[0]),
    aNew(_e->aI, &_s.generators[0]), status(_e->agents.size(),1)
{}

void Client::run()
{
  vector<double> state(env->sI.dim);
  int iAgent, agentStatus;
  double reward;

  while(true)
  {
    if (comm->recvStateFromApp()) break; //sim crashed

    prepareState(iAgent, agentStatus, reward);
    learner->select(iAgent, sNew, aNew, sOld, aOld, agentStatus, reward);

    debugS("Agent %d: [%s] -> [%s] with [%s] rewarded with %f going to [%s]\n",
        iAgent, sOld._print().c_str(), sNew._print().c_str(),
        aOld._print().c_str(), reward, aNew._print().c_str());
    status[iAgent] = agentStatus;

    if(agentStatus != _AGENT_LASTCOMM) {
      prepareAction(iAgent);
      comm->sendActionToApp();
    } else {
      bool bDone = true; //did all agents reach terminal state?
      for (Uint i=0; i<status.size(); i++)
        bDone = bDone && status[i] == _AGENT_LASTCOMM;
      bDone = bDone || env->resetAll; //or does env end is any terminates?

      if(bDone) {
        comm->answerTerminateReq(-1);
        return;
      }
      else comm->answerTerminateReq(1);
    }
  }
}

void Client::prepareState(int& iAgent, int& istatus, Real& reward)
{
  Rvec recv_state(sNew.sInfo.dim);

  unpackState(comm->getDataState(), iAgent, istatus, recv_state, reward);
  assert(iAgent>=0 && iAgent<static_cast<int>(agents.size()));

  sNew.set(recv_state);
  //agent's s is stored in sOld
  agents[iAgent]->Status = istatus;
  agents[iAgent]->swapStates();
  agents[iAgent]->setState(sNew);
  agents[iAgent]->getOldState(sOld);
  agents[iAgent]->getAction(aOld);
  agents[iAgent]->r = reward;
}

void Client::prepareAction(const int iAgent)
{
  if(iAgent<0) die("Error in iAgent number in Client::prepareAction\n");
  assert(iAgent >= 0 && iAgent < static_cast<int>(agents.size()));
  agents[iAgent]->act(aNew);
  double* const buf = comm->getDataAction();
  for (Uint i=0; i<aI.dim; i++) buf[i] = aNew.vals[i];
}
*/
