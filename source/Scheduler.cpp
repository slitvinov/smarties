//
//  Scheduler.cpp
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#include "Misc.h"
#include "Scheduler.h"

#include <unistd.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cassert>

Master::Master(MPI_Comm _c, Learner*const _l, Environment*const _e, Settings&_s):
  slavesComm(_c), learner(_l), env(_e), aI(_e->aI), sI(_e->sI), agents(_e->agents),
  bTrain(settings.bTrain), nPerRank(_e->nAgentsPerRank), nSlaves(settings.nSlaves),
  saveFreq(settings.saveFreq), inSize(2*sizeof(int)+(1+_e->sI.dim)*sizeof(double)),
  outSize(_e->aI.dim*sizeof(double)), inbuf(alloc(inSize)), outbuf(alloc(outSize)),
  gen(settings.gen), sOld(_e->sI), sNew(_e->sI), aOld(_e->aI, settings.gen),
  aNew(_e->aI, settings.gen), totR(0), reward(0), iter(0)
  {
    //the following Irecv will be sent after sending the action
    MPI_Irecv(inbuf, inOneSize, MPI_BYTE, MPI_ANY_SOURCE, 1, slavesComm, &request);
  }

inline void Master::recvState(const int slave, int& iAgent, int& status, Real& reward)
{
    //printf("Master receives from %d - %d (size %d)...\n", agentId, slave, inOneSize);
    byte* buf = inbuf;

    const int recv_iAgent = *((int*) buf);
    iAgent = (slave-1) * nPerRank + recv_iAgent;
    assert(iAgent >= 0 && iAgent < agents.size());
    buf += sizeof(int);


    status = *((int*) buf);
    agents[iAgent]->Status = status;
    buf += sizeof(int);

    sNew.unpack(buf);
    buf += sI.dim * sizeof(double);

    //agent's s is stored in sOld
    agents[iAgent]->swapStates();
    agents[iAgent]->setState(sNew);
    agents[iAgent]->getState(sOld);
    agents[iAgent]->getAction(aOld);

    reward = *((double*) buf);
    agents[iAgent]->r = reward;
    buf += sizeof(double);

    assert(buf-inbuf == inSize);

    MPI_Irecv(inbuf, inOneSize, MPI_BYTE, MPI_ANY_SOURCE, 1, slavesComm, &request);
}

inline void Master::sendAction(const int slave, const int iAgent)
{
    assert(iAgent >= 0 && iAgent < agents.size());
    agents[iAgent]->act(aNew);
    aNew.pack(outbuf);
    MPI_Send(outbuf, outOneSize, MPI_BYTE, slave, 0, slavesComm);
}

void Master::restart(string fname)
{
    learner->restart(fname);
    FILE * f = fopen("master.status", "r");
    int iter_fake(-1);
    if (f == NULL) return;
    fscanf(f, "master iter: %d\n", &iter_fake);
    if(iter_fake>=0) iter = iter_fake;
    printf("master iter: %d\n", iter);
    fclose(f);
}

void Master::save()
{
    ofstream filestats;
    filestats.open("master_rewards.dat", ios::app);
    filestats<<iter<<" "<<totR<<endl;
    filestats.close();
    printf("Iter %d, Reward: %f\n", iter, totR);
    totR = 0.;

    FILE * f = fopen("master.status", "w");
    if (f != NULL) fprintf(f, "master iter: %d\n", iter);
    fclose(f);
    //printf("master iter: %d\n", iter);

    learner->save("policy");
}

void Master::run()
{
    MPI_Status status;
    int completed=0, agentStatus=0, agent;

    while (true) {
        while (true) {
            //if threaded, check on the guys, synchronize apply gradient
            if (nThreads > 1 && learner->checkBatch()) return;

            MPI_Test(&request, &completed, &status);
            if (completed) break;

            //if single thread master, process a batch
            if (nThreads == 1) learner->TrainBatch();
        }

        const int slave = status.MPI_SOURCE;
        recvState(slave, agent, agentStatus, reward);

        if (agentStatus == 3) {
          learner->clearFailedSim((slave-1)*nPerRank, slave*nPerRank);
          continue;
        }

        learner->select(agent, sNew, aNew, sOld, aOld, agentStatus, reward);
        /*
         printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",
         agent, sOld.print().c_str(), sNew.print().c_str(), aOld.print().c_str(), reward, aNew.print().c_str());
         fflush(0);
         */
        if (info != 1) totR += r;
        if (info != 2)  //if terminal, no action required
        sendAction(slave, agent);

        if (++iter % saveFreq == 0) save();
    }
    die("How on earth could you possibly get here? \n");
}

Slave::Slave(Environment*const _env,int _me, Settings& settings):
slavesComm(comm), env(_env), agents(env->agents), me(_me),
bTrain(settings.bTrain), bWriteToFile(!(settings.samplesFile=="none"))
{
}

void Slave::run()
{
  vector<double> status(nStates);
  int iAgent, status;
  double reward;

  while(true) {
      // flag = -1: failed comm, 0: normal, 2: ended
      if (comm->recvStateFromApp()) return; //sim probably crashed

      comm->unpackState(iAgent, status, state, reward);
      comm->sendStateMPI();

      if(status != _AGENT_LASTCOMM)
      {
        comm->recvActionMPI();
        comm->sendActionToApp();
      }
  }
}

void Slave::unpackData(const int iAgent)
{
    byte* cbuf = inbuf;
    actions[iAgent].unpack(cbuf);
}


void Slave::packData(const int iAgent)
{
    byte* cbuf = outbuf;
    const int sSize = env->sI.dim * sizeof(Real);
    const int aSize = env->aI.dim * sizeof(Real);

    *((int*)cbuf) = (int)info[iAgent];
    cbuf += sizeof(int);

    oldStates[iAgent].pack(cbuf);
    cbuf += sSize;

    actions[iAgent].pack(cbuf);
    cbuf += aSize;

    *((Real*)cbuf) = agents[iAgent]->getReward();
    cbuf += sizeof(Real);

    States[iAgent].pack(cbuf);
}

void Slave::restart(string fname)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    FILE * f = fopen(("slave_"+to_string(rank)+".status").c_str(), "r");
    if (f == NULL) return;
    int sim_id_fake = -1;
    fscanf(f, "sim number: %d\n", &sim_id_fake);
    if(sim_id_fake>=0) env->iter = sim_id_fake;
    printf("sim number: %lu\n", env->iter);
    fclose(f);
}

void Slave::save() const
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    FILE * f = fopen(("slave_"+to_string(rank)+".status").c_str(), "w");
    if (f != NULL) fprintf(f, "sim number: %lu\n", env->iter);
    fclose(f);
    //printf( "sim number: %d\n", env->iter);
}

string Slave::bufferTransition(const int iAgent) const
{
    ostringstream o;
    if (not bWriteToFile) return o.str();

    o << iAgent << " " << info[iAgent] << " ";
    o << oldStates[iAgent].printClean().c_str();
    o << States[iAgent].printClean().c_str();
    o << actions[iAgent].printClean().c_str();
    o << agents[iAgent]->getReward();
    return o.str();
}

Client::Client(Learner*const _learner, Environment* const _env, Settings& settings):
  learner(_learner), env(_env), actInfo(_env->aI), sInfo(_env->sI),
  agents(_env->agents),  sOld(_env->sI), s(_env->sI),
  aOld(_env->aI, settings.gen), a(_env->aI, settings.gen), r(0)
{
  learner->restart(settings.restart);

  while(true) {
      int iAgent;
      // flag = -1: failed comm, 0: normal, 2: ended
      const int extflag = env->getState(iAgent);


      agents[iAgent]->getState(s);
      agents[iAgent]->getAction(aOld);
      agents[iAgent]->getOldState(sOld);
      r = agents[iAgent]->getReward();

      //printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",
      //extflag, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(),r, a.print().c_str());

      learner->select(iAgent, s, a, sOld, aOld, extflag, r);

      printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",
      extflag, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(),r, a.print().c_str());

      if(extflag<0) die("Communication lost.\n");

      if(extflag==2) { //then we do not recv an action, we reset
          if(env->resetAll) { //does this env require a full restart upon failing?
              return;
          }
          bool bDone = true;
          for (int i=0; i<agents.size(); i++)
            bDone = bDone && agents[i]->getStatus() == 2;
          if(bDone) return;
      }

      agents[iAgent]->act(a);
      env->setAction(iAgent);
  }
}
