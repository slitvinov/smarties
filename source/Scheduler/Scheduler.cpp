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


Master::Master(Learner* learner, ActionInfo actInfo, StateInfo sInfo) :
learner(learner), actInfo(actInfo), sInfo(sInfo)
{
    inOneSize = 2*sInfo.dim*sizeof(double) + actInfo.dim*sizeof(double) + sizeof(double);
    insize = 1;
    inbuf  = new byte[insize];
    
    outOneSize = actInfo.dim*sizeof(double);
    outsize = 1;
    outbuf  = new byte[outsize];
    
    totR = 0;
}

void Master::execSavers(double time, int iter)
{
    for (auto s : savers)
        if (s->isReady(time, iter)) s->exec();
}

void Master::registerSaver(Saver* saver)
{
    savers.push_back(saver);
    saver->setMaster(this);
}

inline void Master::unpackChunk(byte* &buf, State& sOld, Action& a, double& r, State& s)
{
    // Packing order
    // sOld, a, r, s
    static const int sSize   = sInfo.dim   * sizeof(double);
    static const int actSize = actInfo.dim * sizeof(double);

    sOld.unpack(buf);
    buf += sSize;
    
    a.unpack(buf);
    buf += actSize;

    r = *((double*) buf);
    buf += sizeof(double);
    
    s.unpack(buf);
    buf += sSize;
}

inline void Master::packChunk(byte* &buf, Action a)
{
    // a
    static const int actSize = actInfo.dim * sizeof(double);
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
    Action a(actInfo);
    double r;
    
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

        debug1("Idling time of the master: %d, fetch trials: %d\n", relaxTime, trials);

        debug("Master will receive %d chunks from proc %d of total size %d... ", n, status.MPI_SOURCE, n*inOneSize);
        int slave = status.MPI_SOURCE;
        
        if (n > insize)
        {
            insize = 1.5*n;
            delete[] inbuf;
            inbuf = new byte[insize * inOneSize];
        }
        
        if (n > outsize)
        {
            outsize = 1.5*n;
            delete[] outbuf;
            outbuf = new byte[outsize * outOneSize];
        }
        
        MPI_Recv(inbuf, n*inOneSize, MPI_BYTE, slave, 2, MPI_COMM_WORLD, &status);
        debug("completed\n");
        
        byte* cInbuf = inbuf;
        byte* cOutbuf = outbuf;
        for (int i=0; i<n; i++)
        {
            unpackChunk(cInbuf, sOld, a, r, s);
            
            if (r > -1e10)
            {
                learner->update(sOld, a, r, s);
                debug1("Updating: %s --> %s with %s was rewarded with %f\n", sOld.print().c_str(), s.print().c_str(), a.print().c_str(), r);
                learner->selectAction(s, a);
                totR += r;

                debug1("Chose action %s to send to agent %3d of slave %d\n", a.print().c_str(), i, status.MPI_SOURCE);
            }
            packChunk(cOutbuf, a);
        }

        MPI_Send(outbuf, n*outOneSize, MPI_BYTE, slave, 0, MPI_COMM_WORLD);
        debug("Master sends %d bytes to proc %d\n", n*outOneSize, slave);
        
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

Slave::Slave(Environment* env, double dt, int me) : env(env), agents(env->agents), dt(dt), me(me)
{
    MPI_Request req;
    
    for (int i=0; i<agents.size(); i++)
    {
        actions.push_back(*(new Action(env->aI)));
        oldStates.push_back(*(new State(env->sI)));
    }
    
    insize  = agents.size() * ( env->aI.dim*sizeof(double) );
    outsize = agents.size() * ( 2*env->sI.dim*sizeof(double) + env->aI.dim*sizeof(double) + sizeof(double) );
    
    inbuf  = new byte[insize];
    outbuf = new byte[outsize];
    
    ActionIterator aIter(env->aI);
    RNG* rng = new RNG(rand());
    for (int i=0; i<agents.size(); i++)
    {
        actions[i] = aIter.getRand(rng);
    }
    
    needToPack = new bool[agents.size()];
    
    byte* tmpBuf = new byte[insize];
    byte* cbuf = tmpBuf;
    
    // a
    static const int actSize = env->aI.dim * sizeof(double);
    
    for (int i=0; i<agents.size(); i++)
    {
        actions[i].pack(cbuf);
        cbuf += actSize;
    }

    MPI_Isend(tmpBuf, insize, MPI_BYTE, me, 0, MPI_COMM_WORLD, &req);
}

void Slave::evolve(double& t)
{
    MPI_Request inreq, outreqN, outreqData;
    MPI_Status  status;
	int n = agents.size();
    
    MPI_Recv(inbuf, insize, MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
    debug("Slave %d receives %d bytes\n", me, insize);

    unpackData();

    for (int i = 0; i<n; i++)
	{
        agents[i]->getState(oldStates[i]);
        needToPack[i] = false;
    }

    // TODO: not all of the agents act at the same moment of time
    bool acted = false;
    
    while (!acted)
    {
        for (int i = 0; i<n; i++)
        {
            Agent* agent = agents[i];
            if (agent->getType() != IDLER && agent->getType() != DEAD && t - agent->getLastLearned() > agent->getLearningInterval())
            {
                acted = true;
                debug1("Agent %3d of slave %d will act %s\n", i, me, actions[i].print().c_str());
                agent->act(actions[i]);
                agent->setLastLearned(t);
                needToPack[i] = true;
            };
            
            agent->move(dt);
        }
        
        env->evolve(t);
        t += dt;
    }

    packData();

    MPI_Send(&n, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(outbuf, outsize, MPI_BYTE, 0, 2, MPI_COMM_WORLD);
    debug("Slave %d sends %d chunks of total size %d bytes\n", me, n, outsize);

}

void Slave::unpackData()
{
    byte* cbuf = inbuf;
    
    // a
    static const int actSize = env->aI.dim * sizeof(double);
    
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
    static const int sSize   = env->sI.dim   * sizeof(double);
    static const int actSize = env->aI.dim * sizeof(double);
    
    for (int i=0; i<agents.size(); i++)
    {
        Agent* agent = agents[i];
        
        oldStates[i].pack(cbuf);
        cbuf += sSize;

        actions[i].pack(cbuf);
        cbuf += actSize;
        
        if (needToPack[i])
            *((double*)cbuf) = agent->getReward();
        else
            *((double*)cbuf) = -2e10;
        
        cbuf += sizeof(double);
        
        agent->getState(tmpState);
        tmpState.pack(cbuf);
        cbuf += sSize;
    }
}







