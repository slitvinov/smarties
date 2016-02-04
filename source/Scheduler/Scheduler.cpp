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
    inOneSize = 2*sInfo.dim*sizeof(Real) + actInfo.dim*sizeof(Real) + (1+nInfo)*sizeof(Real);
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

inline void Master::unpackChunk(byte* &buf, State& sOld, Action& a, Real& r, State& s)
{
    // Packing order
    // sOld, a, r, s
    static const int sSize   = sInfo.dim   * sizeof(Real);
    static const int actSize = actInfo.dim * sizeof(Real);
    
    sOld.unpack(buf);
    buf += sSize;
    
    a.unpack(buf);
    buf += actSize;
    
    r = *((Real*) buf);
    buf += sizeof(Real);
    
    s.unpack(buf);
    buf += sSize;
}

inline void Master::unpackChunk(byte* &buf, State& sOld, Action& a, Real& r, vector<Real>& info, State& s)
{
    // Packing order
    // sOld, a, r, s
    static const int sSize   = sInfo.dim   * sizeof(Real);
    static const int actSize = actInfo.dim * sizeof(Real);
    
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
    MPI_Request request;
    MPI_Status  status;
    State sOld(sInfo);
    State s(sInfo);
    Action a(actInfo), aOld(actInfo);
    vector<Real> info;
    Real r;
    
    int n;
    int iter = 0;
    int relaxTime = 10;
    
    debug("Master starting...\n");
    
    while (true)
    {
        MPI_Irecv(&n, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &request);
        int completed;
        int trials = 0;
        MPI_Test(&request, &completed, &status);
        
        while (completed == 0)
        {
            usleep(relaxTime);
            MPI_Test(&request, &completed, &status);
            trials++;
        }
        if (trials < 2 && relaxTime > 1) relaxTime /= 2;
        if (trials > 5) relaxTime *= 2;
        
        debug5("Idling time of the master: %d, fetch trials: %d\n", relaxTime, trials);
        
        debug5("Master will receive %d chunks from proc %d of total size %d... ", n, status.MPI_SOURCE, n*inOneSize);
        int slave = status.MPI_SOURCE;
        
        if (n > insize)
        {
            insize = 1.5*n;
            delete[] inbuf;
            inbuf = new byte[insize * inOneSize];
            printf("This should not be happening: size of inbuf increased.\n");
        }
        
        if (n > outsize)
        {
            outsize = 1.5*n;
            delete[] outbuf;
            outbuf = new byte[outsize * outOneSize];
            printf("This should not be happening: size of outbuf increased.\n");
        }
        
        MPI_Recv(inbuf, n*inOneSize, MPI_BYTE, slave, 2, MPI_COMM_WORLD, &status);
        debug5("completed\n");
        
        byte* cInbuf = inbuf;
        byte* cOutbuf = outbuf;

        for (int i=0; i<n; i++)
        {
            unpackChunk(cInbuf, sOld, aOld, r, info, s);

            //for(int i=0; i<sInfo.dim; i++) debug7("Transition from state %f (n %d) to %f (n %d)\n", sOld.vals[i], _discretize(sOld.vals[i], sOld.sInfo.bottom[i], sOld.sInfo.top[i], sOld.sInfo.bounds[i], sOld.sInfo.belowBottom[i], sOld.sInfo.aboveTop[i]), s.vals[i], _discretize(s.vals[i], s.sInfo.bottom[i], s.sInfo.top[i], s.sInfo.bounds[i], s.sInfo.belowBottom[i], s.sInfo.aboveTop[i]));
            
            if (r > -1e10)
            {
                int agentId = (status.MPI_SOURCE-1) * nAgents + i;
                
                debug4("To learner %d: %s --> %s with %s was rewarded with %f \n", agentId, sOld.print().c_str(), s.print().c_str(), aOld.print().c_str(), r);
                
                Q->passData(agentId, sOld, aOld, s, r, info);

                traces[agentId].add(sOld, aOld);
                learner->updateSelect(traces[agentId], s, a, sOld, aOld, r, agentId);

                totR += r;

                debug4("Chose action %s to send to agent %3d of slave %d\n", a.print().c_str(), i, status.MPI_SOURCE);
            }
            packChunk(cOutbuf, a);
        }
        
        MPI_Send(outbuf, n*outOneSize, MPI_BYTE, slave, 0, MPI_COMM_WORLD);
        debug5("Master sends %d bytes to proc %d\n", n*outOneSize, slave);
        
        // TODO: Add savers
        // TODO: also to the slave processes
        
        iter++;
        if (iter % settings.saveFreq == 0)
        {
            _info("Reward: %f\n", getTotR());
            learner->savePolicy(Saver::folder + "policy");
        }
        
        execSavers(iter * settings.dt, iter);
    }
    
}

Slave::Slave(Environment* env, Real dt, int me) : env(env), agents(env->agents), dt(dt), me(me), first(true), nInfo(env->nInfo)
{
    MPI_Request req;
    
    for (int i=0; i<agents.size(); i++)
    {
        actions.push_back(*(new Action(env->aI)));
        oldStates.push_back(*(new State(env->sI)));
    }
    
    insize  = agents.size() * ( env->aI.dim*sizeof(Real) );
    //G outsize = agents.size() * ( 2*env->sI.dim*sizeof(Real) + env->aI.dim*sizeof(Real) + sizeof(Real) );
    
    outsize = agents.size() * ( 2*env->sI.dim*sizeof(Real) + env->aI.dim*sizeof(Real) + (1+nInfo)*sizeof(Real) );
    
    inbuf  = new byte[insize];
    outbuf = new byte[outsize];
    
    ActionIterator aIter(env->aI);
    RNG* rng = new RNG(rand());
    for (int i=0; i<agents.size(); i++)
        actions[i] = aIter.getRand(rng);
    
    needToPack = new bool[agents.size()];
    
    byte* tmpBuf = new byte[insize];
    byte* cbuf = tmpBuf;
    
    // a
    static const int actSize = env->aI.dim * sizeof(Real);
    
    for (int i=0; i<agents.size(); i++)
    {
        actions[i].pack(cbuf);
        cbuf += actSize;
    }
    
    MPI_Isend(tmpBuf, insize, MPI_BYTE, me, 0, MPI_COMM_WORLD, &req);
}

int Slave::evolve(Real& t)
{
    MPI_Request inreq, outreqN, outreqData;
    MPI_Status  status;
    int n = agents.size();

    if(first)
    {
        for (int i = 0; i<n; i++)
        {
            Agent* agent = agents[i];
            actions[i].vals[0] = env->aI.zeroact;
            agent->act(actions[i]);
        }
        int extflag = env->evolve(t);
        debug6("Slave %d performing first action %d\n", me,env->aI.zeroact);
        //if(extflag) return extflag;
    }
    else // real action, we have done at least one MPI_Send
    {
        MPI_Recv(inbuf, insize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
        debug6("Slave %d receives %d bytes\n", me, insize);
    }
    
    unpackData();
    
    for (int i = 0; i<n; i++)
    {
        agents[i]->getState(oldStates[i]);
        //debug1("Old states %s\n", oldStates[i].print().c_str());
        needToPack[i] = false;
    }
    
    // TODO: not all of the agents act at the same moment of time
    bool acted = false;
    int extflag=0;

    for (int i = 0; i<n; i++)
    {
        Agent* agent = agents[i];
        acted = true;
        if(first)
        {
            actions[i].initAct();
            actions[i].vals[0] = env->aI.zeroact;
            debug4("First random action for agent %3d of slave %d will act %s\n", i, me, actions[i].print().c_str());
        }
        debug6("Agent %3d of slave %d will act %s\n", i, me, actions[i].print().c_str());
        agent->act(actions[i]);
        agent->setLastLearned(t);
        needToPack[i] = true;
        
        agent->move(dt);
    }
    first = false;
    extflag = env->evolve(t);
    t += dt;
    
    packData();
    // if dummy action comment has worked, here we are giving state old, action, new state, reward
    MPI_Send(&n, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(outbuf, outsize, MPI_BYTE, 0, 2, MPI_COMM_WORLD);
    debug6("Slave %d sends %d chunks of total size %d bytes\n", me, n, outsize);
    if (extflag!=0)
    {
        first = true;
        MPI_Recv(inbuf, insize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
        debug6("Slave %d restarts before receiving %d bytes\n", me, insize);
    }
    return 0;
    
}

void Slave::unpackData()
{
    byte* cbuf = inbuf;

    static const int actSize = env->aI.dim * sizeof(Real);
    for (int i=0; i<agents.size(); i++)
    {
        actions[i].unpack(cbuf);
        cbuf += actSize;
    }
}


void Slave::packData()
{
    State tmpState(env->sI);
    byte* cbuf = outbuf;
    
    // Packing order
    // sOld, a, r, s
    static const int sSize   = env->sI.dim * sizeof(Real);
    static const int actSize = env->aI.dim * sizeof(Real);
    
    for (int i=0; i<agents.size(); i++)
    {
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
