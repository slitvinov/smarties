//
//  smarties
//  Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
//  Distributed under the terms of the “CC BY-SA 4.0” license.
//
//  Created by Guido Novati (novatig@ethz.ch).
//

#include "Scheduler.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

Master::Master(MPI_Comm _c, const vector<Learner*> _l, Environment*const _e,
  Settings&_s): workersComm(_c), learners(_l), env(_e), aI(_e->aI), sI(_e->sI),
  agents(_e->agents), bTrain(_s.bTrain), nPerRank(_e->nAgentsPerRank),
  nWorkers(_s.nWorkers), nThreads(_s.nThreads), learn_rank(_s.learner_rank),
  learn_size(_s.learner_size), totNumSteps(_s.totNumSteps),
  outSize(_e->aI.dim*sizeof(double)), inSize((3+_e->sI.dim)*sizeof(double)),
  bAsync(_s.threadSafety>=MPI_THREAD_MULTIPLE),
  inpBufs(alloc_bufs(inSize,nWorkers)), outBufs(alloc_bufs(outSize,nWorkers))
{
  profiler = new Profiler();
  for (Uint i=0; i<learners.size(); i++)  learners[i]->profiler = profiler;
  profiler->stop_start("SLP");

  for(const auto& L : learners) // Figure out if I have on-pol learners
    bNeedSequentialTasks = bNeedSequentialTasks || L->bNeedSequentialTrain();

  if(nWorkers*nPerRank != static_cast<int>(agents.size()))
    die("Mismatch in master's nWorkers nPerRank nAgents.")
  //the following Irecv will be sent after sending the action
  for(int i=1; i<=nWorkers; i++) recvBuffer(i);
}

int Master::run()
{
  if(!bTrain) {
    vector<std::thread> worker_replies = asyncReplyWorkers();
    assert(worker_replies.size() == (size_t) nWorkers);
    for(int i=0; i<nWorkers; i++) worker_replies[i].join();
    return 0;
  }

  while (true)
  {
    if( stepNum >= totNumSteps ) return 0;

    //this is the last possible time to finish the blocking mpi MPI_Allreduce
    // and finally perform the actual gradient step
    for(const auto& L : learners) L->applyGradient();

    //Spawn threads asynchronously handling requests from workers
    vector<std::thread> worker_replies = asyncReplyWorkers();
    assert(worker_replies.size() == (size_t) nWorkers);

    for(const auto& L : learners) L->spawnTrainTasks_par();

    if(bNeedSequentialTasks) {
      // typically on-policy learning. Wait for all needed data:
      for(int i=0; i<nWorkers; i++) worker_replies[i].join();
      // and then perform on-policy update step(s):
      for(const auto& L : learners) L->spawnTrainTasks_seq();
    }

    for(const auto& L : learners) L->prepareGradient();

    if(not bNeedSequentialTasks) {
      //for off-policy learners this is last possibility to wait for needed data
      for(int i=0; i<nWorkers; i++) worker_replies[i].join();
    }

    if(iternum++ % 1000 == 0) flushRewardBuffer();
  }
  die(" ");
  return 1;
}

void Master::processWorker(const int worker)
{
  assert(worker>0 && worker <= (int) nWorkers);
  while(1)
  {
    // If !bTrain this is only termination condition. If bTrain the code checks
    // termination at beginning of loop in run() to prevent undefined behaviors.
    if(!bTrain && seqNum >= totNumSteps) break;
    if( bTrain && learnersLockQueue()  ) break;

    MPI_Status mpistatus;
    int completed = testBuffer(worker, mpistatus);

    if(completed) {
      assert(worker == mpistatus.MPI_SOURCE);
      processAgent(worker, mpistatus);
    } else usleep(5);
  }
}

void Master::processAgent(const int worker, const MPI_Status mpistatus)
{
  //read from worker's buffer:
  vector<double> recv_state(sI.dim);
  int recv_agent = -1, recv_status = -1; double reward;
  unpackState(inpBufs[worker-1], recv_agent, recv_status, recv_state, reward);

  const int agent = (worker-1) * nPerRank + recv_agent;
  Learner*const aAlgo = pickLearner(agent, recv_agent);

  if (recv_status == FAIL_COMM) //app crashed :sadface:
  { //TODO fix for on-pol & multiple algos
    aAlgo->clearFailedSim((worker-1)*nPerRank, worker*nPerRank);
    for(int i=(worker-1)*nPerRank; i<worker*nPerRank; i++) agents[i]->reset();
    warn("Received a FAIL_COMM\n");
  }
  else
  {
    agents[agent]->update(recv_status, recv_state, reward);
    //pick next action and ...do a bunch of other stuff with the data:
    aAlgo->select(*agents[agent]);

    debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]",
      agent, agents[agent]->Status, agents[agent]->sOld->_print().c_str(),
      agents[agent]->s->_print().c_str(), agents[agent]->r,
      agents[agent]->a->_print().c_str());

    sendBuffer(worker, agent);

    if ( recv_status >= TERM_COMM )
    {
      dumpCumulativeReward(agent, aAlgo->iter(), readTimeSteps() );
      #pragma omp atomic
      seqNum++;
    }
    else if ( aAlgo->iter() )
    {
      #pragma omp atomic
      stepNum++;
    }
  }

  recvBuffer(worker);
}

Worker::Worker(Communicator_internal*const _c, Environment*const _e, Settings& _s): comm(_c), env(_e), bTrain(_s.bTrain), status(_e->agents.size(),1) {}

void Worker::run()
{
  //vector<double> state(env->sI.dim);
  //int iAgent, info;
  //double reward;

  while(true) {

    while(true) {
      if (comm->recvStateFromApp()) break; //sim crashed
      //unpackState(comm->getDataState(), iAgent, info, state, reward);
      //status[iAgent] = info;
      //assert(info not_eq FAIL_COMM); //that one should cause the break

      if ( comm->sendActionToApp() ) {
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
