//
//  Scheduler.cpp
//  rl
//
//  Created by Dmitry Alexeev on 22.05.14.
//
//

#include "Scheduler.h"


Master::Master(Learner* learner, ActionInfo actInfo, StateInfo sInfo) :
learner(learner), actInfo(actInfo), sInfo(sInfo)
{
    inOneSize = 2*sInfo.dim*sizeof(double) + actInfo.dim*sizeof(double) + sizeof(double);
    insize = 1;
    inbuf  = new byte[insize];
    
    outOneSize = actInfo.dim*sizeof(double);
    outsize = 1;
    outbuf  = new byte[outsize];
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
    
    while (true)
    {
        MPI_Recv(&n, 1, MPI_BYTE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        
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
        
        MPI_Recv(inbuf, n*inOneSize, MPI_BYTE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
        
        byte* cInbuf = inbuf;
        byte* cOutbuf = outbuf;
        for (int i=0; i<n; i++)
        {
            unpackChunk(cInbuf, sOld, a, r, s);
            learner->update(sOld, a, r, s);
            learner->selectAction(s, a);
            packChunk(cOutbuf, a);
        }
        
        MPI_Isend(outbuf, n*inOneSize, MPI_BYTE, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &request);
        
        // TODO: Add savers
        // TODO: also to the slave processes
    }

}

Slave::Slave(System& newSystem, double newDt) : system(newSystem), agents(newSystem.agents), dt(newDt)
{
    for (int i=0; i<agents.size(); i++)
    {
        actions.push_back(*(new Action(system.actInfo)));
        oldStates.push_back(*(new State(system.sInfo)));
    }
    
    insize  = agents.size() * ( system.actInfo.dim*sizeof(double) );
    outsize = agents.size() * ( 2*system.sInfo.dim*sizeof(double) + system.actInfo.dim*sizeof(double) + sizeof(double) );
    
    inbuf  = new byte[insize];
    outbuf = new byte[outsize];
}

void Slave::evolve(double& t)
{
    MPI_Request inreq, outreqN, outreqData;
    MPI_Status  status;
	int n = agents.size();
    
    MPI_Recv(inbuf, insize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
    
    unpackData();
    
    for (int i = 0; i<n; i++)
	{
        agents[i]->getState(oldStates[i]);
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
                agent->act(actions[i]);
                agent->setLastLearned(t);
            };
            
            agent->move(dt);
        }
        
        system.env->evolve(t);
        t += dt;
    }
    
    packData();
    
    MPI_Isend(&n, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD, &outreqN);
    MPI_Isend(outbuf, outsize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &outreqData);
}

void Slave::unpackData()
{
    byte* cbuf = inbuf;
    
    // a
    int actSize = system.actInfo.dim * sizeof(double);
    
    for (int i=0; i<agents.size(); i++)
    {
        actions[i].unpack(cbuf);
        cbuf += actSize;
    }
}


void Slave::packData()
{
    State tmpState(system.sInfo);
    byte* cbuf = outbuf;
    
    // Packing order
    // sOld, a, r, s
    static const int sSize   = system.sInfo.dim   * sizeof(double);
    static const int actSize = system.actInfo.dim * sizeof(double);
    
    for (int i=0; i<agents.size(); i++)
    {
        Agent* agent = agents[i];
        
        oldStates[i].pack(cbuf);
        cbuf += sSize;
        
        actions[i].pack(cbuf);
        cbuf += actSize;
        
        *((double*)cbuf) = agent->getReward();
        cbuf += sizeof(double);
        
        agent->getState(tmpState);
        tmpState.pack(cbuf);
        cbuf += sSize;
    }
}







