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

Master::Master(Learner* learner, QApproximator* newQ, Environment* env, int nSlaves, Real traceDecay) :
learner(learner), actInfo(env->aI), sInfo(env->sI), nAgents(env->agents.size()), nSlaves(nSlaves), Q(newQ)
{
    nInfo = env->nInfo;
    printf("Adding %d info\n",nInfo);
    
    inOneSize = sizeof(int) + 2*sInfo.dim*sizeof(Real) + actInfo.dim*sizeof(Real) + (1+nInfo)*sizeof(Real);
    //cout << inOneSize << endl;
    insize = 1;
    inbuf  = new byte[insize*inOneSize];
    
    outOneSize = actInfo.dim*sizeof(Real);
    outsize = 1;
    outbuf  = new byte[outsize*outOneSize];
    
    int len;
    if (traceDecay < 1e-10)
        len = 2;
    else
        len = round(-10/log10(traceDecay)) + 1;
    
    traces.resize(nAgents*nSlaves);
    for (auto& t : traces)
    {
        t.len = len;
        t.start = 0;
        t.hist.resize(len);
        for (auto& e : t.hist)
        {
            e.value = -1;
            e.s = new State(sInfo);
            e.a = new Action(actInfo);
        }
    }
    
    totR = 0;
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
    
    first = *((Real*) buf);
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
    int n, iter(0), completed, slave, first;
    
    debug("Master starting...\n");
    
    while (true)
    {
#ifndef MEGADEBUG
        MPI_Irecv(&n, 1, MPI_INT, MPI_ANY_SOURCE, 121, MPI_COMM_WORLD, &request);
        MPI_Test(&request, &completed, &status);
#endif
        bool test = true;
        while(completed == 0)
        {
            if(test)
            {
                Q->Train();
                test=false;
            }
            //debug2("Master trains\n");
            
#ifndef MEGADEBUG
            MPI_Test(&request, &completed, &status);
#endif
        }
        //while (completed == 0);
        
        //debug5("Idling time of the master: %d, fetch trials: %d\n", relaxTime, trials);
        
#ifndef MEGADEBUG
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
                int agentId = max(0,slave*nAgents + i -1);
                
                Q->passData(agentId, first, sOld, aOld, s, r, info);
                
                //traces[agentId].add(sOld, aOld);
                learner->updateSelect(traces[agentId], s, a, sOld, aOld, r, agentId);

                totR += r;
                //printf("To learner %d: %s --> %s with %s was rewarded with %f going to %s \n", agentId, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(), r, a.print().c_str());
                fflush(stdout);
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

Slave::Slave(Environment* env, Real dt, int me) : env(env), agents(env->agents), dt(dt), me(me), first(true), nInfo(env->nInfo)
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
    //cout << outsize << endl;
    
    inbuf  = new byte[insize];
    outbuf = new byte[outsize];
    
    ActionIterator aIter(env->aI);
    RNG* rng = new RNG(rand());
    for (int i=0; i<agents.size(); i++)
        actions[i] = aIter.getRand(rng);
    
    needToPack = new bool[agents.size()];
}

void Slave::evolve(Real& t)
{
#ifndef MEGADEBUG
    MPI_Request inreq, outreqN, outreqData;
    MPI_Status  status;
#endif
    int n = agents.size();
    if(first) //get initial conditions
    {
        for (int i = 0; i<n; i++)
        actions[i].vals[0] = env->aI.zeroact;
        
        int extflag = env->init();
        debug9("Sheduler slave init\n");
        if (extflag<0)
        {
            env->close_Comm();
            env->setup_Comm();
            return;
        }
    }
    
    for (int i = 0; i<n; i++) //backup old states before evolve
    {
        agents[i]->getState(oldStates[i]);
        //debug1("Old states %s\n", oldStates[i].print().c_str());
        needToPack[i] = false;
    }
    
    // TODO: not all of the agents act at the same moment of time
    for (int i = 0; i<n; i++) //perform the action that we got during previous MPI_Recv
    {
        Agent* agent = agents[i];

        debug9("Agent %3d of slave %d will act %s\n", i, me, actions[i].print().c_str());
        agent->act(actions[i]);
        agent->setLastLearned(t);
        needToPack[i] = true;
        
        agent->move(dt);
    }
    
    int extflag = env->evolve(t);
    t += dt;
    
    packData();
    
    if(extflag==0) first = false;
    else           first = true;
    
    //here we are giving state old, action, new state, reward
#ifndef MEGADEBUG
    MPI_Send(&n, 1, MPI_INT, 0, 121, MPI_COMM_WORLD);
    MPI_Send(outbuf, outsize, MPI_BYTE, 0, 2, MPI_COMM_WORLD);
    debug3("Slave %d sends %d chunks of total size %d bytes\n", me, n, outsize);
    MPI_Recv(inbuf, insize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
#endif
    unpackData();
    
    if (extflag<0)
    {
        env->close_Comm();
        env->setup_Comm();
    }
}

void Slave::unpackData()
{
    byte* cbuf = inbuf;

    static const int actSize = env->aI.dim * sizeof(Real);
    for (int i=0; i<agents.size(); i++)
    {
        actions[i].unpack(cbuf);
        cbuf += actSize;
        //debug9("Agent %3d of slave will act %s\n", i, actions[i].print().c_str());
    }
}

void Slave::packData()
{
    State tmpState(env->sI);
    byte* cbuf = outbuf;
    
    // Packing order
    // first, sOld, a, r, info, s
    static const int sSize   = env->sI.dim * sizeof(Real);
    static const int actSize = env->aI.dim * sizeof(Real);
    
    for (int i=0; i<agents.size(); i++)
    {
        *((Real*)cbuf) = first;
        cbuf += sizeof(int);
        
        Agent* agent = agents[i];
        
        oldStates[i].pack(cbuf);
        cbuf += sSize;
        
        actions[i].pack(cbuf);
        cbuf += actSize;
        
        if (needToPack[i])
            *((Real*)cbuf) = agent->getReward();
        else
            *((Real*)cbuf) = -2e10;
        
        cbuf += sizeof(Real);

        for (int i = 0; i<nInfo; i++)
        {
            if (needToPack[i])
                *((Real*)cbuf) = agent->getInfo(i);
            else
                *((Real*)cbuf) = -2e10;
            cbuf += sizeof(Real);
        }
        
        agent->getState(tmpState);
        tmpState.pack(cbuf);
        cbuf += sSize;
    }
}
