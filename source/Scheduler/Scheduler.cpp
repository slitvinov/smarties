//
//  Scheduler.cpp
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#include "../ErrorHandling.h"
#include "Scheduler.h"
#include <unistd.h>
#include <iostream>
#include "../Misc.h"
#include <iomanip>
#include <string>
#include <fstream>
#include <algorithm>
#include <cassert>
//#define TRAINING
Master::Master(Learner* learner, Environment* env, Settings & settings) :
learner(learner), actInfo(env->aI), sInfo(env->sI), nAgents(env->agents.size()), nSlaves(settings.nSlaves), bTRAINING(settings.bTrain==1), totR(0.0)
{
    nInfo = env->nInfo;
    
    inOneSize = sizeof(int) + 2*sInfo.dim*sizeof(Real) + actInfo.dim*sizeof(Real) + (1+nInfo)*sizeof(Real);
    insize = 1;
    inbuf  = new byte[insize*inOneSize];
    
    outOneSize = actInfo.dim*sizeof(Real);
    outsize = 1;
    outbuf  = new byte[outsize*outOneSize];
}

void Master::execSavers(Real time, int iter)
{
    for (auto s : savers)
        if (s->isReady(time, iter)) s->exec();
}

void Master::registerSaver(Saver* saver)
{
    savers.push_back(saver);
    saver->setMaster(this);
}

inline void Master::unpackChunk(byte* &buf, int & first, State& sOld, Action& a, Real& r, vector<Real>& info, State& s)
{
    // Packing order
    // sOld, a, r, s
    static const int sSize   = sInfo.dim   * sizeof(Real);
    static const int actSize = actInfo.dim * sizeof(Real);
    
    first = *((int*) buf);
    buf += sizeof(int);
    
    sOld.unpack(buf);
    buf += sSize;
    
    a.unpack(buf);
    buf += actSize;
    
    r = *((Real*) buf);
    buf += sizeof(Real);
    
    //G
    for (int i = 0; i<nInfo; i++)
    {
        info[i] = *((Real*) buf);
        buf += sizeof(Real);
    }
    
    s.unpack(buf);
    buf += sSize;
}

inline void Master::packChunk(byte* &buf, Action a)
{
    static const int actSize = actInfo.dim * sizeof(Real);
    a.pack(buf);
    buf += actSize;
}

void Master::restart(string fname)
{
    learner->try2restart(fname);
}

void Master::run()
{
#ifndef MEGADEBUG
    MPI_Request request;
    MPI_Status  status;
#endif
    State  sOld(sInfo),   s(sInfo);
    Action aOld(actInfo), a(actInfo);
    vector<Real> info(nInfo);
    Real r;
    int n, iter, completed(0), slave, first;
    
    debug("Master starting...\n");
    
    while (true)
    {
        if (bTRAINING)
        {
        #ifndef MEGADEBUG
            MPI_Irecv(&n, 1, MPI_INT, MPI_ANY_SOURCE, 121, MPI_COMM_WORLD, &request);
            MPI_Test(&request, &completed, &status);
        #endif
            while(completed == 0)
            {
                learner->Train();
        #ifndef MEGADEBUG
                MPI_Test(&request, &completed, &status);
        #endif
            }
        }
    #ifndef MEGADEBUG
        else MPI_Recv(&n, 1, MPI_INT, MPI_ANY_SOURCE, 121, MPI_COMM_WORLD, &status);
        
    
        debug3("Master will receive %d chunks from proc %d of total size %d... ", n, status.MPI_SOURCE, n*inOneSize);
        slave = status.MPI_SOURCE;
    #endif
        
        if (n > insize)
        {
            insize = 1.5*n;
            delete[] inbuf;
            inbuf = new byte[insize * inOneSize];
            printf("This should not be happening? size of inbuf increased.\n");
        }
        
        if (n > outsize)
        {
            outsize = 1.5*n;
            delete[] outbuf;
            outbuf = new byte[outsize * outOneSize];
            printf("This should not be happening? size of outbuf increased.\n");
        }
        
    #ifndef MEGADEBUG
        MPI_Recv(inbuf, n*inOneSize, MPI_BYTE, slave, 2, MPI_COMM_WORLD, &status);
        debug5("completed\n");
    #endif
        byte* cInbuf = inbuf;
        byte* cOutbuf = outbuf;
        
        for (int i=0; i<n; i++)
        {
            unpackChunk(cInbuf, first, sOld, aOld, r, info, s);
            if (r > -1e10)
            {
                int agentId = slave*nAgents +i-1;
                assert(agentId>=0);
                learner->T->passData(agentId, first, sOld, aOld, s, r, info);
                learner->updateSelect(agentId, s, a, sOld, aOld, info, r);
                printf("To learner %d: %s --> %s with %s was rewarded with %f going to %s\n", agentId, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(), r, a.print().c_str());
                totR += r;
            }
            packChunk(cOutbuf, a);
        }
#ifndef MEGADEBUG
        MPI_Send(outbuf, n*outOneSize, MPI_BYTE, slave, 0, MPI_COMM_WORLD);
        debug9("Master sends %d bytes to proc %d\n", n*outOneSize, slave);
#endif
        
        // TODO: Add savers, also to the slave processes
        iter++;
        if (iter % settings.saveFreq == 0)
        {
            printf("Reward: %f\n", getTotR());
            //learner->savePolicy(Saver::folder + "policy");
        }
        
        //execSavers(iter * settings.dt, iter);
#ifdef MEGADEBUG
        abort();
#endif
    }
    
}

Slave::Slave(Environment* env, int me, Settings & settings) :
env(env), agents(env->agents), dt(settings.dt), me(me), first(true), nInfo(env->nInfo), bTRAINING(settings.bTrain==1)
{
#ifndef MEGADEBUG
    MPI_Request req;
#endif
    for (int i=0; i<agents.size(); i++)
    {
        actions.push_back(*(new Action(env->aI)));
        oldStates.push_back(*(new State(env->sI)));
    }
    
    insize  = agents.size() * ( env->aI.dim*sizeof(Real) );
    outsize = agents.size() * ( sizeof(int) + 2*env->sI.dim*sizeof(Real) + env->aI.dim*sizeof(Real) + (1+nInfo)*sizeof(Real) );
    
    inbuf  = new byte[insize];
    outbuf = new byte[outsize];
    
    ActionIterator aIter(env->aI);
    rng = new RNG(rand());
    
    //will be used for async actions
    needToPack = new bool[agents.size()];
}

void Slave::evolve(Real& t)
{
#ifndef MEGADEBUG
    MPI_Request inreq, outreqN, outreqData;
    MPI_Status  status;
#endif
    int n = agents.size();
    if(first)
    {
        debug3("Getting initial conditions for slave %d.\n",me);
        int extflag = env->init();
        
        if (bTRAINING)
        {
            if (extflag<0)
            {  //if comm failed, just retry and pray client wakes up
                env->close_Comm();
                env->setup_Comm();
                return;
            }
            if (extflag>0)
            { //then we restarted a failed sim or smth, start anew
                return;
            }
            
            //random first action
            ActionIterator aIter(env->aI);
            for (int i = 0; i<n; i++) actions[i] = aIter.getRand(rng);
        }
        else
        {
            if (extflag!=0) die("Simulation is over?\n");
            for (int i = 0; i<n; i++)
            {
                agents[i]->getState(oldStates[i]);
                learner->updateSelect(0, oldStates[i], actions[i], oldStates[i], actions[i], agents[i]->Info, 0); //TODO, for now not training...
                //actions[i].vals[0] = env->aI.zeroact;
            }
        }
    }
    
    // TODO: not all of the agents act at the same moment of time
    for (int i = 0; i<n; i++)
    {
        //backup old states before evolve
        agents[i]->getState(oldStates[i]);
        //perform the action that we got during previous MPI_Recv
        agents[i]->act(actions[i]);
        debug3("Agent %d of slave %d was in %s and will act %s.\n", i, me, oldStates[i].print().c_str(), actions[i].print().c_str());
        
        agents[i]->setLastLearned(t);
        agents[i]->move(dt);
        needToPack[i] = true; //will be used for ansync actions
    }
    
    int extflag = env->evolve(t);
    t += dt; //scary: damn dipoles
    
    if (bTRAINING && extflag<0)
    {
        first = true;
        env->close_Comm();
        env->setup_Comm();
        return;
    }
    else if (not bTRAINING && extflag!=0) die("Simulation is over?\n");
    packData();
    
    //here we are passing state old, action, new state, reward to master
#ifndef MEGADEBUG
    MPI_Send(&n, 1, MPI_INT, 0, 121, MPI_COMM_WORLD);
    MPI_Send(outbuf, outsize, MPI_BYTE, 0, 2, MPI_COMM_WORLD);
    debug3("Slave %d sends %d chunks of total size %d bytes\n", me, n, outsize);
    
    //just dumping all info for safety
    ofstream fout;
    fout.open(("obs"+to_string(me)+".dat").c_str(),ios::app);
    for (int i = 0; i<n; i++)
    {
        State tmpState(env->sI);
        Agent* agent = agents[i];
        agent->getState(tmpState);
        
        fout << i << " " << first << " ";
        fout << oldStates[i].printClean().c_str();
        fout << tmpState.printClean().c_str();
        fout << actions[i].printClean().c_str();
        fout << agent->getReward();
        for (int i = 0; i<agent->nInfo; i++)
        fout << " " << agent->getInfo(i);
        fout << endl;
    }
    fout.close();
    
    //recv the action
    MPI_Recv(inbuf, insize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
#endif
    
    if(extflag==0) first = false;
    else           first = true;
    unpackData();
}

void Slave::unpackData()
{
    byte* cbuf = inbuf;

    static const int actSize = env->aI.dim * sizeof(Real);
    for (int i=0; i<agents.size(); i++)
    {
        actions[i].unpack(cbuf);
        cbuf += actSize;
        debug9("Agent %3d of slave will act %s\n", i, actions[i].print().c_str());
    }
}

void Slave::packData()
{
    State tmpState(env->sI);
    byte* cbuf = outbuf;
    
    // Packing order: first, sOld, a, r, info, s
    static const int sSize   = env->sI.dim * sizeof(Real);
    static const int actSize = env->aI.dim * sizeof(Real);
    
    for (int i=0; i<agents.size(); i++)
    {
        *((int*)cbuf) = (int)first;
        cbuf += sizeof(int);
        
        Agent* agent = agents[i];
        
        oldStates[i].pack(cbuf);
        cbuf += sSize;
        
        actions[i].pack(cbuf);
        cbuf += actSize;
        
        //if (needToPack[i])
        *((Real*)cbuf) = agent->getReward();
        //else
        //*((Real*)cbuf) = -2e10;
        
        cbuf += sizeof(Real);

        for (int k = 0; k<nInfo; k++)
        {
           //if (needToPack[i])
           *((Real*)cbuf) = agent->getInfo(k);
           //else
           //*((Real*)cbuf) = -2e10;
           cbuf += sizeof(Real);
        }
    
        agent->getState(tmpState);
        tmpState.pack(cbuf);
        cbuf += sSize;
    }
}

void Slave::restart(string fname) //TODO
{
    learner->try2restart(fname);
}
