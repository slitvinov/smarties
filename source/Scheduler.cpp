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

Master::Master(Learner* learner, Environment* env, Settings & settings) :
learner(learner), actInfo(env->aI), sInfo(env->sI), bTrain(settings.bTrain==1),
nAgents(env->agents.size()), nSlaves(settings.nSlaves), saveFreq(settings.saveFreq),
iter(0), agentId(-1), gen(settings.gen), sOld(sInfo), s(sInfo), aOld(actInfo,gen),
a(actInfo,gen), totR(0), r(0), requested(false)
{
    inOneSize = sizeof(int) + (2*env->sI.dim +env->aI.dim +1)*sizeof(Real);
    inbuf  = new byte[inOneSize];
    
    outOneSize = actInfo.dim*sizeof(Real);
    outbuf  = new byte[outOneSize];
}

inline void Master::unpackChunk(byte* buf, int & info, State& sOld, Action& a, Real& r, State& s)
{
    static const int sSize   = sInfo.dim   * sizeof(Real);
    static const int actSize = actInfo.dim * sizeof(Real);
    
    info = *((int*) buf);
    buf += sizeof(int);
    
    sOld.unpack(buf);
    buf += sSize;
    
    a.unpack(buf);
    buf += actSize;
    
    r = *((Real*) buf);
    buf += sizeof(Real);
    
    s.unpack(buf);
}

inline void Master::packChunk(byte* buf, Action a)
{
    //static const int actSize = actInfo.dim * sizeof(Real);
    a.pack(buf);
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
    printf( "master iter: %d\n", iter);
    
    learner->save("policy");
}

void Master::run()
{
    int agentId(0), completed(0), slave(0), info(0);
    debug("Master starting...\n");
    
    while (true) {
        if (bTrain) {
            #ifndef MEGADEBUG
            MPI_Irecv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &request);
            #endif
            while (true) {
                learner->TrainBatch();
                #ifndef MEGADEBUG
                MPI_Test(&request, &completed, &status);
                if (completed == 1) break;
                #endif
            }
        }
        #ifndef MEGADEBUG
        else MPI_Recv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        slave = status.MPI_SOURCE;
        //printf("Master receives from %d - %d (size %d)...\n", agentId, slave, inOneSize);
        MPI_Recv(inbuf, inOneSize, MPI_BYTE, slave, 2, MPI_COMM_WORLD, &status);
        #endif
        unpackChunk(inbuf, info, sOld, aOld, r, s);
        const int agentID = (slave)*nAgents +agentId;
        assert(agentID>=0);
        learner->select(agentID, s, a, sOld, aOld, info, r);
        //printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",agentID,
        //sOld.print().c_str(),s.print().c_str(),aOld.print().c_str(),r,a.print().c_str());
        totR += r;
        if (info != 2) { //not terminal
            #ifndef MEGADEBUG
            packChunk(outbuf, a);
            MPI_Send(outbuf, outOneSize, MPI_BYTE, slave, 0, MPI_COMM_WORLD);
            #endif
        }
        if (iter++ % saveFreq == 0) save();
    }
}

void Master::hustle()
{
    int completed(0), slave(0), info(0);
    #ifndef MEGADEBUG
    if(not requested) {
        MPI_Irecv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &request);
        requested = true;
    }
    #endif
    
    debug("Master starting... (completed? %d)\n",completed);
    while (true) {
        while (true) {
            #ifndef MEGADEBUG
            MPI_Test(&request, &completed, &status);
            if (completed == 1) break;
            #endif
            if (learner->checkBatch()) return;
        }
        #ifndef MEGADEBUG
        slave = status.MPI_SOURCE;
        
        //printf("Master receives from %d - %d (size %d)...\n", agentId, slave, inOneSize);
        
        MPI_Recv(inbuf, inOneSize, MPI_BYTE, slave, 2, MPI_COMM_WORLD, &status);
        #endif
        unpackChunk(inbuf, info, sOld, aOld, r, s);
        const int agentID = (slave)*nAgents +agentId;
        learner->select(agentID, s, a, sOld, aOld, info, r);
        
        //printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",agentID,
        //sOld.print().c_str(),s.print().c_str(),aOld.print().c_str(),r,a.print().c_str());
        
        totR += r;
        if (info != 2) { //not terminal
            #ifndef MEGADEBUG
            packChunk(outbuf, a);
            MPI_Send(outbuf, outOneSize, MPI_BYTE, slave, 0, MPI_COMM_WORLD);
            #endif
        }
        if (iter++ % saveFreq == 0) save();
        
        #ifndef MEGADEBUG
        MPI_Irecv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &request);
        #endif
    }
}
/*
void Master::hustle()
{
    int agentId(0), completed(0), slave(0), info(0);
    while (true) {
        #ifndef MEGADEBUG
        MPI_Recv(&agentId, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        slave = status.MPI_SOURCE;
        //printf("Master receives from %d - %d (size %d)...\n", agentId, slave, inOneSize);
        MPI_Recv(inbuf, inOneSize, MPI_BYTE, slave, 2, MPI_COMM_WORLD, &status);
        #endif
        unpackChunk(inbuf, info, sOld, aOld, r, s);
        const int agentID = (slave)*nAgents +agentId;
        assert(agentID>=0);
        learner->select(agentID, s, a, sOld, aOld, info, r);
        //printf("To learner %d: %s --> %s with %s rewarded with %f going to %s\n",agentID,
        //sOld.print().c_str(),s.print().c_str(),aOld.print().c_str(),r,a.print().c_str());
        totR += r;
        if (info != 2) { //not terminal
            #ifndef MEGADEBUG
            packChunk(outbuf, a);
            MPI_Send(outbuf, outOneSize, MPI_BYTE, slave, 0, MPI_COMM_WORLD);
            #endif
        }
        if (iter++ % saveFreq == 0) save();
        if (learner->checkBatch()) return;
    }
}
*/
Slave::Slave(Environment* env, int me, Settings & settings) :
env(env), agents(env->agents), me(me), bTrain(settings.bTrain)
{
    #ifndef MEGADEBUG
    MPI_Request req;
    #endif
    info.resize(agents.size());
    for (int i=0; i<agents.size(); i++) {
        actions.push_back(*(new Action(env->aI, settings.gen)));
        oldStates.push_back(*(new State(env->sI)));
        States.push_back(*(new State(env->sI)));
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
    MPI_Status status;
    MPI_Request req;
    #endif
    for (int i(0); i<info.size(); i++) info[i] = 1;
    int iAgent;
    while(true)
    {
        // flag = -1: failed comm, 0: normal, 2: ended
        const int extflag = env->getState(iAgent);
             if (    bTrain &&  extflag<0) return; //if comm failed, retry & pray
        else if (not bTrain && (extflag<0 || extflag>1)) die("Simulation is over?\n");
        
        agents[iAgent]->getState(States[iAgent]);
        agents[iAgent]->getAction(actions[iAgent]);
        agents[iAgent]->getOldState(oldStates[iAgent]);

        #ifndef MEGADEBUG
        if (extflag==2) {
            const bool proper = info[iAgent]==0;
            
            info[iAgent] = 2;
            packData(iAgent);
            MPI_Send(&iAgent, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Isend(outbuf, outsize, MPI_BYTE, 0, 2, MPI_COMM_WORLD, &req);
            
            if(proper) {
                ofstream fout;
                fout.open(("obs"+to_string(me)+".dat").c_str(),ios::app);
                fout << bufferTransition(iAgent) << endl;
                fout.close();
            }
            
            if(env->resetAll) {
                save();
                for (int i(0); i<info.size(); i++) info[i] = 1;
            }
            else info[iAgent] = 1;
        } else { //simulation ended, no need for action
            packData(iAgent);
            MPI_Send(&iAgent, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Isend(outbuf, outsize, MPI_BYTE, 0, 2, MPI_COMM_WORLD, &req);
            
            if(info[iAgent]==0) {
                ofstream fout;
                fout.open(("obs"+to_string(me)+".dat").c_str(),ios::app);
                fout << bufferTransition(iAgent) << endl;
                fout.close();
            }
            info[iAgent] = 0;
            
            MPI_Recv(inbuf, insize, MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
            unpackData(iAgent);
            agents[iAgent]->act(actions[iAgent]);
            env->setAction(iAgent);
            debug3("Agent %d of slave %d was in %s and will act %s.\n",
                   iAgent, me, States[iAgent].print().c_str(), actions[iAgent].print().c_str());
        }
        #endif
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
    //learner->restart(fname);
    FILE * f = fopen(("sim_"+to_string(me)+".status").c_str(), "r");
    if (f == NULL) return;
    int sim_id_fake = -1;
    fscanf(f, "sim number: %d\n", &sim_id_fake);
    if(sim_id_fake>=0) env->iter = sim_id_fake;
    printf("sim number: %d\n", env->iter);
    fclose(f);
}

void Slave::save() const
{
    FILE * f = fopen(("sim_"+to_string(me)+".status").c_str(), "w");
    if (f != NULL) fprintf(f, "sim number: %d\n", env->iter);
    fclose(f);
    printf( "sim number: %d\n", env->iter);
}

string Slave::bufferTransition(const int iAgent) const
{
    ostringstream o;
    o << iAgent << " " << info[iAgent] << " ";
    o << oldStates[iAgent].printClean().c_str();
    o << States[iAgent].printClean().c_str();
    o << actions[iAgent].printClean().c_str();
    o << agents[iAgent]->getReward();
    return o.str();
}
