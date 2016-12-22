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

Master::Master(MPI_Comm comm, Learner*const _learner, Environment* const _env,
  Settings& settings): slavesComm(comm), learner(_learner), env(_env), actInfo(_env->aI),
  sInfo(_env->sI), bTrain(settings.bTrain==1), nAgents(_env->agents.size()),
  nSlaves(settings.nSlaves), saveFreq(settings.saveFreq), iter(0),
  gen(settings.gen), sOld(_env->sI), s(_env->sI), aOld(_env->aI, settings.gen),
  a(_env->aI, settings.gen), totR(0), r(0), requested(false)
{
    inOneSize = sizeof(int) + (2*env->sI.dim +env->aI.dim +1)*sizeof(Real);
    inbuf  = new byte[inOneSize];

    outOneSize = actInfo.dim*sizeof(Real);
    outbuf  = new byte[outOneSize];
}

inline void Master::unpackChunk(byte* buf, int & info, State& _sOld, Action& _a, Real& _r, State& _s)
{
    static const int sSize   = sInfo.dim   * sizeof(Real);
    static const int actSize = actInfo.dim * sizeof(Real);

    info = *((int*) buf);
    buf += sizeof(int);

    _sOld.unpack(buf);
    buf += sSize;

    _a.unpack(buf);
    buf += actSize;

    _r = *((Real*) buf);
    buf += sizeof(Real);

    _s.unpack(buf);
}

inline void Master::packChunk(byte* buf, Action _a)
{
    //static const int actSize = actInfo.dim * sizeof(Real);
    _a.pack(buf);
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
    int agentId(0), completed(0), info(0);
    //printf("Master starting...\n");
    MPI_Status  status;

    while (true) {
        #ifndef MEGADEBUG
        MPI_Irecv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &request);
        #endif
        while (true) {
            #ifndef MEGADEBUG
            MPI_Test(&request, &completed, &status);
            if (completed == 1) break;
            #endif
            learner->TrainBatch();
        }
        #ifndef MEGADEBUG

        const int slave = status.MPI_SOURCE - 1, srcID = status.MPI_SOURCE;
        //printf("Master receives from %d - %d (size %d)...\n", agentId, slave, inOneSize);
        MPI_Recv(inbuf, inOneSize, MPI_BYTE, srcID, 2, MPI_COMM_WORLD, &status);

        unpackChunk(inbuf, info, sOld, aOld, r, s);
        if (info == 3) {
          learner->clearFailedSim(slave*nAgents, (slave+1)*nAgents);
          continue;
        }
        const int agentID = (slave)*nAgents +agentId;
        assert(agentID>=0);
        learner->select(agentID, s, a, sOld, aOld, info, r);
        /*
         printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",
         agentID, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(),r, a.print().c_str());
         fflush(0);
         */
        if (info != 1) totR += r;
        if (info != 2) { //not terminal
            packChunk(outbuf, a);
            MPI_Send(outbuf, outOneSize, MPI_BYTE, srcID, 0, MPI_COMM_WORLD);
        }

        if (++iter % saveFreq == 0) save();
        #endif
    }
}

void Master::hustle()
{
#ifndef MEGADEBUG
    MPI_Status  status;
    int completed(0), info(0), cnt(0), knt(0), agentId(0);

    //this is only called the first time, the following Irecv will be sent at the end of the first comm
    //the idea is that while the communication is not done, we continue processing the NN update
    if(not requested) {
        MPI_Irecv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &request);
        requested = true;
    }

    while (true) {
        while (true) {
            MPI_Test(&request, &completed, &status);
            if (completed == 1) {
                cnt++;
                break;
            }
            //if true: finish processing the dqn update and come right back to hustling
            if (learner->checkBatch())  return;
            knt++;
        }

        const int slave = status.MPI_SOURCE - 1, srcID = status.MPI_SOURCE;
        //printf("Master receives from %d - %d (size %d)...\n", agentId, slave, inOneSize);
        MPI_Recv(inbuf, inOneSize, MPI_BYTE, srcID, 2, MPI_COMM_WORLD, &status);

        unpackChunk(inbuf, info, sOld, aOld, r, s);

        if (info == 3) {
          learner->clearFailedSim(slave*nAgents, (slave+1)*nAgents);
          //prepare Recv for next round
          MPI_Irecv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &request);
          continue;
        }
        const int agentID = (slave)*nAgents +agentId;
        assert(agentID>=0);
        learner->select(agentID, s, a, sOld, aOld, info, r);
        /*
         printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",
         agentID, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(),r, a.print().c_str());
         fflush(0);
         */
        if (info != 1) totR += r; //not first state, this is just to track performance
        if (info != 2) { //if terminal, no action required
            packChunk(outbuf, a);
            MPI_Send(outbuf, outOneSize, MPI_BYTE, srcID, 0, MPI_COMM_WORLD);
        }

        if (++iter % saveFreq == 0) save();
        //prepare Recv for next round
        MPI_Irecv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &request);
    }
#endif
    die("How on earth could you possibly get here? \n");
}

Slave::Slave(MPI_Comm comm, Environment*const _env,int _me, Settings& settings):
slavesComm(comm), env(_env), agents(env->agents), me(_me),
bTrain(settings.bTrain), bWriteToFile(!(settings.samplesFile=="none"))
{
    #ifndef MEGADEBUG
    //MPI_Request req;
    #endif
    info.resize(agents.size());
    for (int i=0; i<agents.size(); i++) {
        actions.push_back(Action(env->aI, settings.gen));
        oldStates.push_back(State(env->sI));
        States.push_back(State(env->sI));
        info[i] = 1;
    }

    insize  = env->aI.dim*sizeof(Real);
    outsize = sizeof(int) + (2*env->sI.dim +env->aI.dim +1)*sizeof(Real);
    inbuf  = new byte[insize];
    outbuf = new byte[outsize];
}

void Slave::run()
{
    #ifndef MEGADEBUG
    for (int i=0; i<info.size(); i++) info[i] = 1;
    int iAgent;
    while(true) {
        // flag = -1: failed comm, 0: normal, 2: ended
        const int extflag = env->getState(iAgent);

        if ( bTrain &&  extflag<0) {
            //printf("\nSIMULATION CRASHED\n\n");
            sendFail(iAgent);
            MPI_Ssend(&iAgent, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Ssend(outbuf, outsize, MPI_BYTE, 0, 2, MPI_COMM_WORLD);

            env->iter++;
            //abort(); //
            return; //if comm failed, retry & pray
        }
        //if (extflag<0) die("Comm failed, call again & pray.\n");
        //not bTrain assumes that im only interested in one run (e.g. for animation)
        if (not bTrain && extflag==2) die("Simulation is over.\n");

        agents[iAgent]->getState(States[iAgent]);
        agents[iAgent]->getAction(actions[iAgent]);
        agents[iAgent]->getOldState(oldStates[iAgent]);

        std::string outBuffer;
        //if i observed at least one transition, fill a buffer to write to file
        if(info[iAgent]==0) {
            if (extflag==2) info[iAgent] = 2;
            outBuffer = bufferTransition(iAgent);
        }

        packData(iAgent);
        MPI_Ssend(&iAgent, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Ssend(outbuf, outsize, MPI_BYTE, 0, 2, MPI_COMM_WORLD);

        //if buffer is not empty, print it to file, while we wait for action
        if (!outBuffer.empty()) {
            ofstream fout;
            fout.open(("obs"+to_string(me)+".dat").c_str(),ios::app);
            fout << outBuffer << endl;
            fout.close();
        }

        if(info[iAgent]==2) { //then we do not recv an action, we reset
            if(env->resetAll) { //does this env require a full restart upon failing?
                save();
                for (int i(0); i<info.size(); i++) info[i] = 1;
            }
            else info[iAgent] = 1; //else i just restart one agent (e.g. multiple carts env?)
        } else {
            MPI_Status status;
            MPI_Recv(inbuf, insize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
            unpackData(iAgent);
            agents[iAgent]->act(actions[iAgent]);
            /*
             printf("Agent %d of slave %d was in %s and will act %s.\n",
             iAgent, me, States[iAgent].print().c_str(), actions[iAgent].print().c_str());
             fflush(0);
             */
            env->setAction(iAgent);
            //if you got here, then state you send will not be an initial condition
            //(ie. sOld and aOld contain something)
            info[iAgent] = 0;
        }
    }
    #endif
    die("How on earth could you possibly get here? \n");
}

void Slave::unpackData(const int iAgent)
{
    byte* cbuf = inbuf;
    actions[iAgent].unpack(cbuf);
}

void Slave::sendFail(const int iAgent)
{
    byte* cbuf = outbuf;
    *((int*)cbuf) = 3;
    cbuf += sizeof(int);
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
    //learner->restart(fname);
    FILE * f = fopen(("sim_"+to_string(me)+".status").c_str(), "r");
    if (f == NULL) return;
    int sim_id_fake = -1;
    fscanf(f, "sim number: %d\n", &sim_id_fake);
    if(sim_id_fake>=0) env->iter = sim_id_fake;
    printf("sim number: %lu\n", env->iter);
    fclose(f);
}

void Slave::save() const
{
    FILE * f = fopen(("sim_"+to_string(me)+".status").c_str(), "w");
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
  agents(_env->agents), nAgents(_env->agents.size()), sOld(_env->sI), s(_env->sI),
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

      learner->select(iAgent, s, a, sOld, aOld, extflag, r);

      if(info[iAgent]==2) { //then we do not recv an action, we reset
          if(env->resetAll) { //does this env require a full restart upon failing?
              return;
          }
      }

      agents[iAgent]->act(actions[iAgent]);
      env->setAction(iAgent);
      info[iAgent] = 0;
  }
}
