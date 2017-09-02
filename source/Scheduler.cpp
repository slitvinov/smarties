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
  if(nSlaves*nPerRank != static_cast<int>(agents.size()))
    die("Mismatch in master's nSlaves nPerRank nAgents.")
  //the following Irecv will be sent after sending the action
  for(int i=1; i<=nSlaves; i++) {
    MPI_Irecv(inpBufs[i-1], inSize, MPI_BYTE, i, 1, slavesComm, &requests[i-1]);
    slaveIrecvStatus[i-1] = OPEN;
  }
}

int Master::recvState(const int slave)
{
  vector<Real> recv_state(sI.dim);
  int recv_iAgent = -1, istatus;
  double reward;
  unpackState(inpBufs[slave-1], recv_iAgent, istatus, recv_state, reward);
  const int iAgent = (slave-1) * nPerRank + recv_iAgent;
  assert(iAgent>=0 && recv_iAgent>=0 && iAgent<static_cast<int>(agents.size()));
  agents[iAgent]->update(istatus, recv_state, reward);

  if (istatus == _AGENT_LASTCOMM)
  {
    char path[256];
    sprintf(path, "cumulative_rewards_rank%02d.dat", learn_rank);
    std::ofstream outf(path, ios::app);
    outf<<learner->iter()<<" "<<iAgent<<" "<<agents[iAgent]->transitionID<<" "<<agents[iAgent]->cumulative_rewards<<endl;
    outf.close();
  }
  return iAgent;
}

void Master::restart(string fname)
{
  learner->restart(fname);
}

void Master::save()
{
  learner->save("policy");
}

int Master::run()
{
  while (true)
  {
    if(!bTrain && stepNum >= totNumSteps) return 1;
    if( bTrain && learner->reachedMaxGradStep()) return 1;
    profiler->stop_start("PREP");
    learner->prepareData(); //sync data, make sure we can sample
    profiler->stop_start("TASK");

    #pragma omp parallel num_threads(nThreads)
    #pragma omp master
    {
      if(postponed_queue.size())
      {
        profiler->stop_start("QUEUE");
        for (const auto& w : postponed_queue)
        {
          const int slave = w.first, agent = w.second;
          addToNTasks(1);
          #pragma omp task firstprivate(slave, agent)
            processRequest(slave, agent);
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

        for(int i=0; i<nSlaves; i++) // && not learner->batchGradientReady()
        {
          int completed=0;
          MPI_Status mpistatus;
          {
            lock_guard<mutex> lock(mpi_mutex);
            if(slaveIrecvStatus[i] == OPEN) //otherwise, Irecv not sent
              MPI_Test(&requests[i], &completed, &mpistatus);
            else if (slaveIrecvStatus[i] == SEND)
            {
              MPI_Request tmp;
              MPI_Isend(outBufs[i], outSize, MPI_BYTE, i+1, 0, slavesComm,&tmp);
              MPI_Request_free(&tmp); //Not my problem
              debugS("Sent action to slave %d: [%s]", i+1, print(vector<Real>(outBufs[i], outBufs[i]+aI.dim)).c_str());
              MPI_Irecv(inpBufs[i], inSize, MPI_BYTE, i+1, 1, slavesComm, &requests[i]);
              slaveIrecvStatus[i] = OPEN;
            }
            else if(slaveIrecvStatus[i] == OVER)
            {
              MPI_Irecv(inpBufs[i], inSize, MPI_BYTE, i+1, 1, slavesComm, &requests[i]);
              slaveIrecvStatus[i] = OPEN;
            } else assert(slaveIrecvStatus[i] == DOING);
          }

          if(completed)
          {
            int slave = mpistatus.MPI_SOURCE;
            assert(slaveIrecvStatus[i]==OPEN && slave==i+1);
            debugS("Master receives from %d", slave);
            const int agent = recvState(slave); //unpack buffer
            slaveIrecvStatus[slave-1] = DOING; //slave will be 'served' by task

            if(learner->readyForAgent(slave, agent))
            {
              addToNTasks(1);
#pragma omp task firstprivate(slave, agent) if(learner->readNTasks()<nThreads)
              processRequest(slave, agent);
            }
            else postponed_queue.push_back(make_pair(slave, agent));
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

void Master::processRequest(const int slave, const int agent)
{
  const int thrID = omp_get_thread_num();
  assert(agent >= 0 && agent < static_cast<int>(agents.size()));
  if(thrID==1) learner->profiler->check_start("SERV");
  if (agents[agent]->Status == _AGENT_FAILCOMM) //app crashed :sadface:
  {
    //TODO fix for on-pol: on crash clear unfinished workspace assigned to slave
    learner->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
    for (int i=(slave-1)*nPerRank; i<slave*nPerRank; i++) agents[i]->reset();
    printf("Received an _AGENT_FAILCOMM\n");
    slaveIrecvStatus[slave-1] = OVER;
  }
  else
  {
    //pick next action and ...do a bunch of other stuff with the data:
    learner->select(agent, *agents[agent]);

    debugS("Agent %d (%d): [%s] -> [%s] rewarded with %f going to [%s]", agent, agents[agent]->Status, agents[agent]->sOld->_print().c_str(), agents[agent]->s->_print().c_str(), agents[agent]->r, agents[agent]->a->_print().c_str());

    if (agents[agent]->Status != _AGENT_LASTCOMM)
    {
      for(Uint i=0; i<aI.dim; i++)
        outBufs[slave-1][i] = agents[agent]->a->vals[i];

      lock_guard<mutex> lock(mpi_mutex);
      slaveIrecvStatus[slave-1] = SEND;
    }
    else
    { //if terminal, no action required
      lock_guard<mutex> lock(mpi_mutex);
      slaveIrecvStatus[slave-1] = OVER;
      //if(env->resetAll) TODO
      //  learner->pushBackEndedSim((slave-1)*nPerRank, slave*nPerRank);
      ++stepNum; //sequence counter: used to terminate if not training
    }
  }
  if(thrID==1) learner->profiler->check_start("SLP");
  addToNTasks(-1);
}

int Master::learnerReadyForAgent(const int slave, const int agent) const
{
  //Return whether we need more data from this agent:
  //generally will return true. Except when on-policy algorithms (ie. GAE).
  //  For example: if batch is almost ready and waiting from the last agents to
  //  finish sequence, getting more data would then be a waste because then the
  //  gradient will be applied, and data would become off-policy and unusable
  //However, if I receive data of a brand new seq from agent B on slave S while
  //waiting for terminal state of agent C, also on slave S, then user is NOT
  //using correct algorithm for the problem or has implemented something wrong.
  //There is a check on on-policy algo to verify that when new seq from agent
  //on a slave S begins, all other slave S's agents must have sent term state.
  return learner->readyForAgent(slave, agent);
  //assert(ready || agents[agent]->Status == _AGENT_FIRSTCOMM); //for on pol
}

Slave::Slave(Communicator*const _c, Environment*const _e, Settings& _s):
        comm(_c), env(_e), bTrain(_s.bTrain), status(_e->agents.size(),1) {}

void Slave::run()
{
  vector<double> state(env->sI.dim);
  int iAgent, agentStatus;
  double reward;

  while(true) {

    while(true) {
      if (comm->recvStateFromApp()) break; //sim crashed
      unpackState(comm->getDataState(), iAgent, agentStatus, state, reward);

      status[iAgent] = agentStatus;
      if(agentStatus != _AGENT_LASTCOMM)
      {
        if (comm->sendActionToApp()) {
          printf("Slave exiting\n");
          fflush(0);
          return;
        }
      } else {
        /*
          bool bDone = true; //did all agents reach terminal state?
          for (Uint i=0; i<status.size(); i++)
            bDone = bDone && status[i] == _AGENT_LASTCOMM;
          bDone = bDone || env->resetAll; //does env end is any terminates?
          if(bDone && !bTrain) {
            comm->answerTerminateReq(-1);
            return;
          }
          else
         */
        comm->answerTerminateReq(1.);
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
