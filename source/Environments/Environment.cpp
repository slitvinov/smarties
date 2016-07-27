/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "Environment.h"
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <cstdio>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>

using namespace std;

Environment::Environment(const int nAgents, const string execpath, const int _rank, Settings & settings) :
execpath(execpath), rank(_rank), g(settings.gen), n(nAgents), resetAll(true), workerid(_rank),
sock(0), ListenerSocket(0), bytes(0), iter(0), max_scale(20, -1000), min_scale(20, 1000), gamma(settings.gamma)
{
    for (int i=0; i<nAgents; i++) agents.push_back(new Agent(i));
}

void Environment::setup_Comm()
{
    workerid = rank;
    
    string dummy = "/tmp/sock_";
    sprintf(SOCK_PATH, "%s%d", dummy.c_str(), workerid);
    printf("mserver: SOCK_PATH=->%s<-\n", SOCK_PATH);
    
    sizein = (2 + sI.dim + 1)*sizeof(double); //nagent, tag, state, reward
    datain = (double *) malloc(sizein);
    sizeout = aI.dim * sizeof(double);
    dataout = (double *) malloc(sizeout);
    
    spawn_server();
    //printf("comm dim = %d %d \n", sizein, sizeout);
    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    
    /* Specify the server */
    bzero((char *)&serverAddress, sizeof(serverAddress));
    serverAddress.sun_family = AF_UNIX;
    strcpy(serverAddress.sun_path, SOCK_PATH);
    const int servlen = sizeof(serverAddress.sun_family) + strlen(serverAddress.sun_path);
    
    /* Connect to the server */
    while (connect(sock, (struct sockaddr *)&serverAddress, servlen) < 0) {
        //perror("connecting...\n");
    }
}

void Environment::close_Comm()
{
    close(ListenerSocket);
}

void Environment::spawn_server()
{
    const int rf = fork();
    if (rf == 0) {
        char line[1024];
        char *largv[64];
        
        mkdir(("simulation_"+to_string(rank)+"/").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        chdir(("simulation_"+to_string(rank)+"/").c_str());
        
        sprintf(line, execpath.c_str());
        parse(line, largv);     // prepare argv
        
        #if 1==0 //if true goes to stdout
        char output[256];
        sprintf(output, "output_%d_%d", workerid,iter);
        int fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        dup2(fd, 1);    // make stdout go to file
        dup2(fd, 2);    // make stderr go to file
        close(fd);      // fd no longer needed
        #endif
        
        //printf("About to exec.... \n");
        cout << execpath << endl << *largv << endl;
        
        //int res = execlp(execpath.c_str(), execpath.c_str(), NULL);
        const int res = execlp(execpath.c_str(), execpath.c_str(),to_string(workerid).c_str(), NULL);
        //int res = execvp(*largv, largv);
        
        //printf("Returning from exec\n");
        if (res < 0) die("Unable to exec file '%s'!\n", execpath.c_str());
    }
    
    //printf("waiting for server to setup everything..\n");
    //sleep(2); //pause is not safe with MPI
    //printf("ok, I continue...\n");
}

void Environment::setAction(const int & iAgent)
{
    for (int i=0; i<aI.dim; i++) {
        dataout[i] = (double) agents[iAgent]->a->valsContinuous[i];
        assert(not std::isnan(agents[iAgent]->a->valsContinuous[i]) && not std::isinf(agents[iAgent]->a->valsContinuous[i]));
    }
    send_all(sock, dataout, sizeout);
}

int Environment::getState(int & iAgent)
{
    int bStatus = 0;
    //printf("RECEIVING %d,%d\n",sock,sizein);
    if ((bytes = recv_all(sock, datain, sizein)) <= 0) {
        if (bytes == 0) printf("socket %d hung up\n", sock);
        else perror("(1) recv");
        
        close(sock);
        bStatus = -1;
    } else { // (bytes == nbyte)
        iAgent  = *((int*)  datain   );
        bStatus = *((int*) (datain+1)); //first (==1?), terminal (==2?), etc
        debug3("Receiving from agent %d %d: ", iAgent, bStatus);
        
        std::swap(agents[iAgent]->s,agents[iAgent]->sOld);
        
        int k = 2;
        for (int j=0; j<sI.dim; j++) {
            debug3(" %f (%d)",datain[k],k);
            agents[iAgent]->s->vals[j] = (Real) datain[k++];
            assert(not std::isnan(agents[iAgent]->s->vals[j]) && not std::isinf(agents[iAgent]->s->vals[j]));
        }
        
        debug3(" %f (%d)\n",datain[k],k);
        agents[iAgent]->r = (Real) datain[k++];
        assert(not std::isnan(agents[iAgent]->r) && not std::isinf(agents[iAgent]->r));
        debug3("Got from child %d: reward %f initial state %s\n", rank, agents[iAgent]->r, agents[iAgent]->s->print().c_str()); fflush(0);
    }
    fflush(0);
    return bStatus;
}

void Environment::setDims() //this environment is for the cart pole test
{
    {
        sI.bounds.clear(); sI.top.clear(); sI.bottom.clear(); sI.isLabel.clear(); sI.inUse.clear();
        // State: coordinate...
        sI.bounds.push_back(12);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        
        // ...velocity...
        sI.bounds.push_back(6);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        
        // ...and angular velocity
        sI.bounds.push_back(6);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        
        // ...angle...
        sI.bounds.push_back(16);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
    }
    {
        aI.realValues = false;
        aI.dim = 1;
        aI.zeroact = 2;
        aI.values.resize(aI.dim);
        
        for (int i=0; i<aI.dim; i++) {
            aI.bounds.push_back(7);
            aI.upperBounds.push_back( 1.);
            aI.lowerBounds.push_back(-1.);
            
            aI.values[i].push_back(-20.);
            aI.values[i].push_back(-5.);
            aI.values[i].push_back(-1.);
            aI.values[i].push_back(0.0);
            aI.values[i].push_back(1.);
            aI.values[i].push_back(5.);
            aI.values[i].push_back(20.);
        }
    }
    commonSetup();
}

void Environment::commonSetup()
{
    assert(sI.bottom.size() == sI.top.size());
    assert(sI.bottom.size() == sI.inUse.size());
    assert(sI.bottom.size() == sI.bounds.size());
    assert(sI.bottom.size() == sI.isLabel.size());
    
    sI.dim = 0; sI.dimUsed = 0;
    for (int i=0; i<sI.bounds.size(); i++) {
        sI.dim++;
        if (sI.inUse[i]) sI.dimUsed++;
    }
    
    aI.shifts.resize(aI.dim);
    aI.shifts[0] = 1;
    for (int i=1; i < aI.dim; i++) {
        assert(aI.bounds[i] == aI.values[i].size());
        aI.shifts[i] = aI.shifts[i-1] * aI.bounds[i-1];
    }
    
    for (auto& a : agents) {
        a->setDims(sI, aI);
        a->a = new Action(aI, g);
        a->s = new State(sI);
        a->sOld = new State(sI);
    }
}

bool Environment::pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward)
{
    bool new_sample(false);
    if (reward<-0.9) new_sample=true;
#ifndef _scaleR_
    reward = 1. - fabs(t_sN.vals[3])/0.2;            //max cumulative reward = sum gamma^t r < 1/(1-gamma)
    if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
#else
    reward = (1. - fabs(t_sN.vals[3])/0.1)*(1.-gamma); //max cumulative reward = sum gamma^t r < 1/(1-gamma) = 1
    if (new_sample) reward = -1.;  // = - max cumulative reward
#endif
    return new_sample; //cart pole has failed if r = -1, need to clean this shit and rely only on info
}
/*
void ExternalEnvironment::setDims()
{
    sI.dim = 6;
    // State: Horizontal distance from goal point...
    sI.bounds.push_back(22); //one block in between the bounds, one more on each side
    sI.top.push_back(1.);
    sI.bottom.push_back(-1.);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false);
    
    // ...vertical distance...
    sI.bounds.push_back(22);
    sI.top.push_back(1.);
    sI.bottom.push_back(-1.);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false);
    
    // ...inclination of the fish...
    sI.bounds.push_back(22); // only positive or negative
    sI.top.push_back(1.5);
    sI.bottom.push_back(-1.5);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false);
    
    // ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
    sI.bounds.push_back(2); // Will get ~ 0 or 0.5
    sI.top.push_back(1.0);
    sI.bottom.push_back(0.0);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(false);
    
    // ...last action (HAX!)
    sI.bounds.push_back(3);
    sI.top.push_back(3.0);
    sI.bottom.push_back(0.0);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(true);
    
    // ...second last action (HAX!)
    sI.bounds.push_back(3);
    sI.top.push_back(3.0);
    sI.bottom.push_back(0.0);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(true);
    
    aI.dim = 1; //How many actions taken per turn by one agent
    
    for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3); //Number of possible actions to choose from (nothing, curve right, curve left)    
    
    aI.values.push_back(0.0);
    aI.values.push_back(2.0);
    aI.values.push_back(-2.);

    sI.values.push_back(0.0);
    sI.values.push_back(2.0);
    sI.values.push_back(-2.);
    nInfo = 2; 
    aI.zeroact = 0;
    for (auto& a : agents)
    {
        a->Info.resize(nInfo);
        a->nInfo = nInfo;
    }
}
*/

/*
void HardCartEnvironment::setDims()
{
    sI.dim = 2;
    printf("Created the correct cart??\n");
    // State: coordinate...
    sI.bounds.push_back(12);
    sI.top.push_back(2.0);
    sI.bottom.push_back(-2.0);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false);
    
    // ...angle...
    sI.bounds.push_back(16);
    sI.top.push_back(0.2);
    sI.bottom.push_back(-0.2);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false);
    
    aI.dim = 1;
    
    for (int i=0; i<aI.dim; i++) aI.bounds.push_back(5);
    
    aI.values.push_back(-2.);
    aI.values.push_back(-.5);
    aI.values.push_back(0.0);
    aI.values.push_back(0.5);
    aI.values.push_back(2.0);
}
*/

/*
 void GlideEnvironment::setDims()
 {
 sI.dim = 6;
 // State: u velocity...
 sI.bounds.push_back(7);
 sI.top.push_back(1);
 sI.bottom.push_back(0);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);
 
 // ...v velocity...
 sI.bounds.push_back(7);
 sI.top.push_back(0);
 sI.bottom.push_back(-1);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);
 
 // ...angular velocity...
 sI.bounds.push_back(7);
 sI.top.push_back(0.5);
 sI.bottom.push_back(-0.5);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);
 
 // ...x pos...
 sI.bounds.push_back(26);
 sI.top.push_back(20);
 sI.bottom.push_back(-100);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);
 
 // ...y pos...
 sI.bounds.push_back(20);
 sI.top.push_back(50.001);
 sI.bottom.push_back(-.001);
 sI.aboveTop.push_back(false);
 sI.belowBottom.push_back(false);
 
 // ...angle...
 sI.bounds.push_back(12);
 sI.top.push_back(3.14159265359);
 sI.bottom.push_back(-3.14159265359);
 sI.aboveTop.push_back(true);
 sI.belowBottom.push_back(true);
 
 // ...torque...
 //sI.bounds.push_back(10);
 //sI.top.push_back(0.8);
 //sI.bottom.push_back(-0.8);
 //sI.aboveTop.push_back(true);
 //sI.belowBottom.push_back(true);

aI.dim = 1;

for (int i=0; i<aI.dim; i++) aI.bounds.push_back(3);
}
*/

