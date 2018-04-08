//
//  Scheduler.cpp
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#include "Scheduler.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <chrono>

Master::Master(MPI_Comm _c, const vector<Learner*> _l, Environment*const _e,
  Settings&_s): slavesComm(_c), learners(_l), env(_e), aI(_e->aI), sI(_e->sI),
  agents(_e->agents), bTrain(_s.bTrain), nPerRank(_e->nAgentsPerRank),
  nSlaves(_s.nSlaves), nThreads(_s.nThreads), learn_rank(_s.learner_rank),
  learn_size(_s.learner_size), totNumSteps(_s.totNumSteps),
  outSize(_e->aI.dim*sizeof(double)), inSize((3+_e->sI.dim)*sizeof(double)),
  inpBufs(alloc_bufs(inSize,nSlaves)), outBufs(alloc_bufs(outSize,nSlaves)),
  requests(nSlaves, MPI_REQUEST_NULL)
{
  profiler = new Profiler();
  profiler_int = learners[0]->profiler;
  for (Uint i=0; i<learners.size(); i++) {
    learners[i]->profiler_ext = profiler;
    if(i) {
      _dispose_object(learners[i]->profiler);
      learners[i]->profiler = profiler_int;
    }
  }

  if(nSlaves*nPerRank != static_cast<int>(agents.size()))
    die("Mismatch in master's nSlaves nPerRank nAgents.")
  //the following Irecv will be sent after sending the action
  for(int i=1; i<=nSlaves; i++) recvBuffer(i);
}

int Master::run()
{
  while (true)
  {
    if( stepNum >= totNumSteps ) return 0;

    profiler->stop_start("PREP");
    prepareLearners();

    profiler->stop_start("SLP");
    #pragma omp parallel num_threads(nThreads)
    {
      #pragma omp single nowait
        for(int i=1; i<=nSlaves; i++)
          #pragma omp task firstprivate(i) priority(1)
            processSlave(i);

      #pragma omp single nowait
        spawnTrainingTasks_par();
    }

    spawnTrainingTasks_seq();

    profiler->stop_start("TERM");
    for(const auto& L : learners)
      L->prepareGradient(); //tasks have finished, update is ready

    if(iternum++ % 1000 == 0) flushRewardBuffer();
  }
  die(" ");
  return 1;
}

void Master::processSlave(const int slave)
{
  if( bTrain && learnersLockQueue() ) return;
  const int thrID = omp_get_thread_num();
  if(thrID==1) profiler_int->stop_start("SERV");
  if(thrID==0) profiler->stop_start("SERV");

  int completed = 0;
  MPI_Status mpistatus;
  {
    assert(slave>0 && slave <= (int) nSlaves);
    lock_guard<mutex> lock(mpi_mutex);
    MPI_Test(&requests[slave-1], &completed, &mpistatus);
  }
  //printf("Thread %d doing slave %d\n",thrID,slave);

  if(completed) {
    assert(slave == mpistatus.MPI_SOURCE);
    processAgent(slave, mpistatus);
  }

  if(thrID==1) profiler_int->stop_start("SLP");
  if(thrID==0) profiler->stop_start("SLP");

  // If not bTrain this is only termination condition. If bTrain the code checks
  // termination at beginning of loop in run() to prevent undefined behaviors.
  if( !bTrain && readTimeSteps() >= (Uint) totNumSteps ) return;
  if(  bTrain && learnersLockQueue() ) return;

  #pragma omp task firstprivate(slave) priority(1)
    processSlave(slave);
}

void Master::processAgent(const int slave, const MPI_Status mpistatus)
{
  //read from slave's buffer:
  vector<double> recv_state(sI.dim);
  int recv_agent = -1, recv_status = -1; double reward;
  unpackState(inpBufs[slave-1], recv_agent, recv_status, recv_state, reward);

  const int agent = (slave-1) * nPerRank + recv_agent;
  Learner*const aAlgo = pickLearner(agent, recv_agent);

  if (recv_status == FAIL_COMM) //app crashed :sadface:
  { //TODO fix for on-pol & multiple algos
    aAlgo->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
    for(int i=(slave-1)*nPerRank; i<slave*nPerRank; i++) agents[i]->reset();
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

    sendBuffer(slave, agent);

    if ( recv_status >= TERM_COMM )
    {
      dumpCumulativeReward(agent, aAlgo->iter(), readTimeSteps() );
    }
    else if ( aAlgo->iter() )
    {
      #pragma omp atomic
      stepNum++;
    }
  }

  recvBuffer(slave);
}

Slave::Slave(Communicator*const _c, Environment*const _e, Settings& _s):
        comm(_c), env(_e), bTrain(_s.bTrain), status(_e->agents.size(),1) {}

void Slave::run()
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
        die("Slave exiting");
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
