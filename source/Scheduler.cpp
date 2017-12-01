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

Master::Master(MPI_Comm _c,Learner*const _l, Environment*const _e, Settings&_s):
  slavesComm(_c),learner(_l),env(_e),aI(_e->aI),sI(_e->sI), agents(_e->agents),
  bTrain(_s.bTrain), nPerRank(_e->nAgentsPerRank), saveFreq(_s.saveFreq),
  nSlaves(_s.nSlaves), nThreads(_s.nThreads), learn_rank(_s.learner_rank),
  learn_size(_s.learner_size), totNumSteps(_s.totNumSteps),
  outSize(_e->aI.dim*sizeof(double)), inSize((3+_e->sI.dim)*sizeof(double)),
  inpBufs(alloc_bufs(inSize,nSlaves)), outBufs(alloc_bufs(outSize,nSlaves)),
  slaveIrecvStatus(nSlaves, EMPTY), agentSortingCheck(agents.size(), 0), requests(nSlaves, MPI_REQUEST_NULL)//, profiler(_l->profiler)
{
  profiler = new Profiler();
  learner->profiler_ext = profiler;
  printf("%d %d %d %d %d %d\n", nThreads, nSlaves, learn_rank, inSize, outSize, learn_size);
fflush(0);
  if(nSlaves*nPerRank != static_cast<int>(agents.size()))
    die("Mismatch in master's nSlaves nPerRank nAgents.")
  //the following Irecv will be sent after sending the action
  for(int i=1; i<=nSlaves; i++) recvBuffer(i);
}

int Master::run()
{
  while (true)
  {
    if(!bTrain && stepNum >= totNumSteps) return 1;
    if( bTrain && learner->reachedMaxGradStep()) return 1;
    profiler->stop_start("PREP");
    learner->prepareData(); //sync data, make sure we can sample

    #pragma omp parallel num_threads(nThreads)
    #pragma omp master
    {
      if(postponed_queue.size() && learner->unlockQueue())
      {
        profiler->stop_start("QUEUE");
        for (const int slave : postponed_queue) {
          addToNTasks(1);
          #pragma omp task firstprivate(slave) //priority(1)
            processRequest(slave);
        }
        postponed_queue.clear();
      }

      bool first = true;
      profiler->stop_start("COMM");
      while (true)
      {
        if(!bTrain && stepNum >= totNumSteps) break; //check for termination
        if(learner->batchGradientReady()) break;

        spawnTrainingTasks(first, postponed_queue.size()==(size_t)nSlaves);

        for(int i=0; i<nSlaves; i++)
        {
          int completed=0;
          MPI_Status mpistatus;
          {
            lock_guard<mutex> lock(mpi_mutex);
            if(slaveIrecvStatus[i] == OPEN) //otherwise, Irecv not sent
              MPI_Test(&requests[i], &completed, &mpistatus);
          }

          if(completed)
          {
            const int slave = mpistatus.MPI_SOURCE;
            assert(slaveIrecvStatus[i]==OPEN && slave==i+1);
            debugS("Master receives from %d", slave);
            slaveIrecvStatus[slave-1] = DOING; //slave will be 'served' by task

            if(learnerReadyForAgent(slave))
            {
              addToNTasks(1);
              #pragma omp task firstprivate(slave) if(readNTasks()<nThreads) //priority(1)
                processRequest(slave);
            }
            else postponed_queue.push_back(slave);
            debugS("number of tasks %d", learner->readNTasks());
            assert(learner->readNTasks()>=0);
          }
        }
      }
    }

    profiler->stop_start("TERM");
    learner->applyGradient(); //tasks have finished, update is ready
  }
  die(" ");
  return 0;
}

inline void Master::processRequest(const int slave)
{
  const int thrID = omp_get_thread_num();
  //printf("Thread %d doing slave %d\n",thrID,slave);
  if(thrID==1) learner->profiler->stop_start("SERV");
  if(thrID==0) profiler->stop_start("SERV");
  //auto start = std::chrono::high_resolution_clock::now();
  vector<Real> recv_state(sI.dim);
  int recv_iAgent = -1, istatus;
  double reward;
  //read from slave's buffer:
  unpackState(inpBufs[slave-1], recv_iAgent, istatus, recv_state, reward);
  const int agent = (slave-1) * nPerRank + recv_iAgent;
  assert(agent>=0 && recv_iAgent>=0 && agent<static_cast<int>(agents.size()));

       if (istatus == FAIL_COMM) //app crashed :sadface:
  {
    //TODO fix for on-pol: on crash clear unfinished workspace assigned to slave
    learner->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
    for (int i=(slave-1)*nPerRank; i<slave*nPerRank; i++) agents[i]->reset();
    printf("Received a FAIL_COMM\n");
  }
  else if (istatus == GAME_OVER)
  {
    //TODO fix for on-pol: on crash clear unfinished workspace assigned to slave
    learner->pushBackEndedSim((slave-1)*nPerRank, slave*nPerRank);
    for (int i=(slave-1)*nPerRank; i<slave*nPerRank; i++) agents[i]->reset();
    //printf("Received a GAME_OVER\n");
  }
  else
  {
    agents[agent]->update(istatus, recv_state, reward);
    assert(istatus == agents[agent]->Status);
    //pick next action and ...do a bunch of other stuff with the data:
    learner->select(*agents[agent]);
    const auto iter = learner->iter();

    debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]", agent, agents[agent]->Status, agents[agent]->sOld->_print().c_str(), agents[agent]->s->_print().c_str(), agents[agent]->r, agents[agent]->a->_print().c_str());

    if(agents[agent]->Status not_eq TERM_COMM) //if term, no action sent
    {
      for(Uint i=0; i<aI.dim; i++)
        outBufs[slave-1][i] = agents[agent]->a->vals[i];
      sendBuffer(slave);
    }
    else if ( iter || !bTrain )
    {
      char path[256];
      sprintf(path, "cumulative_rewards_rank%02d.dat", learn_rank);
      lock_guard<mutex> lock(dump_mutex);
      std::ofstream outf(path, ios::app);
      outf<<iter<<" "<<agent<<" "<<agents[agent]->transitionID<<" "
          <<agents[agent]->cumulative_rewards<<endl;
      outf.close();
      ++stepNum; //sequence counter: used to terminate if not training
    }
  }

  recvBuffer(slave);
  addToNTasks(-1);
  if(thrID==1) learner->profiler->stop_start("SLP");
  if(thrID==0) profiler->stop_start("COMM");
  //auto elapsed = std::chrono::high_resolution_clock::now() - start;
  //cout << chrono::duration_cast<chrono::microseconds>(elapsed).count() <<endl;
}

Slave::Slave(Communicator*const _c, Environment*const _e, Settings& _s):
        comm(_c), env(_e), bTrain(_s.bTrain), status(_e->agents.size(),1) {}

void Slave::run()
{
  vector<double> state(env->sI.dim);
  int iAgent, info;
  double reward;

  while(true) {

    while(true) {
      if (comm->recvStateFromApp()) break; //sim crashed
      unpackState(comm->getDataState(), iAgent, info, state, reward);

      status[iAgent] = info;
      if(info not_eq TERM_COMM && info not_eq GAME_OVER)
      {
        assert(info not_eq FAIL_COMM); //that one should cause the break
        if (comm->sendActionToApp()) {
          printf("Slave exiting\n");
          fflush(0);
          return;
        }
      }
    }
    //if here, a crash happened:
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
  vector<Real> recv_state(sNew.sInfo.dim);

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
