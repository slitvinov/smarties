/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include <sys/types.h>
#include <sys/stat.h>
#include <cstdio>
#include <unistd.h>
#include <string>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>
#include "ExternalEnvironment.h"


using namespace std;

ExternalEnvironment::ExternalEnvironment(vector<Agent*> _agents, string execpath, StateType tp, int _rank) :
Environment(_agents), execpath(execpath), rank(_rank), iter(0), workerid(_rank), callid(0), sock(0), ListenerSocket(0), bytes(0), addr_len(0), servlen(0)
{
    n = agents.size();
    //for (auto a : agents)
    //    agents.push_back(a);
    rewards.resize(n);
    states.resize(n);
    sI.type = tp;
    if (_agents.size()!=n) {
        printf("Something wrong in the constructor\n");
    }
}

void ExternalEnvironment::setup_Comm()
{
    string dummy = "/tmp/sock_";
    sprintf(SOCK_PATH, "%s%d", dummy.c_str(), workerid);
    printf("mserver: SOCK_PATH=->%s<-\n", SOCK_PATH);
    //signal(SIGUSR2, sighandler);// not safe with MPI
    
    probdim = n*(sI.dim + 1 + nInfo); //state, reward, additional stuff
    sizein = probdim*sizeof(double);
    datain = (double *) malloc(sizein);
    sizeout = n*sizeof(double);
    dataout = (double *) malloc(sizeout);
    
    spawn_server(0);
    
    printf("problem dim = %i %d %d %d \n", probdim, n, sizein, sizeout);
    printf("starting...\n");
    
    sock = socket(AF_UNIX, SOCK_STREAM, 0);
    
    /* Specify the server */
    bzero((char *)&serverAddress, sizeof(serverAddress));
    serverAddress.sun_family = AF_UNIX;
    strcpy(serverAddress.sun_path, SOCK_PATH);
    servlen = sizeof(serverAddress.sun_family) + strlen(serverAddress.sun_path);
    
    /* Connect to the server */
    while (connect(sock, (struct sockaddr *)&serverAddress, servlen) < 0)
    {
        //perror("connecting...\n");
    }
    printf("CONNECTED?!?! %d\n",sock);
    fflush(0);
    //exit(1);	// here, we can sleep and retry
}

void ExternalEnvironment::close_Comm()
{
    //close(ListenerSocket);
}

void ExternalEnvironment::spawn_server(int worker_id)
{
    int rf = fork();
    if (rf == 0)
    {
        char line[1024];
        char *largv[64];
        
        mkdir(("simulation_"+to_string(rank)+"/").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        chdir(("simulation_"+to_string(rank)+"/").c_str());
        
        sprintf(line, execpath.c_str());
        parse(line, largv);     /* prepare argv */
        
        #if 1==1 //else goes to stdout
        char output[256];
        sprintf(output, "output_%d_%d", workerid,iter++);
        int fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        dup2(fd, 1);    // make stdout go to file
        dup2(fd, 2);    // make stderr go to file
        close(fd);      // fd no longer needed
        #endif
        
        printf("About to exec.... \n");
        cout << execpath << endl;
        cout << *largv << endl;
        
        //int res = execlp(execpath.c_str(), execpath.c_str(), NULL);
        int res = execlp(execpath.c_str(), execpath.c_str(),to_string(workerid).c_str(), NULL);
        //int res = execvp(*largv, largv);
        
        printf("Returning from exec\n");
        if (res < 0) die("Unable to exec file '%s'!\n", execpath.c_str());
    }
    
    printf("waiting for server to setup everything..\n");
    //sleep(2);
    //	pause();	// signals + MPI is not a safe solution
    //	we could check a file here 
    printf("ok, I continue...\n");
}

int ExternalEnvironment::evolve(Real t)
{
    bStatus = 0;
    bool _nan = false;
    for (int i=0; i<n; i++)
    {
        dataout[i] = (double) agents[i]->a->vals[0];
        printf("Client rank %d sent child %d: action %f\n", rank, i, dataout[i]);
    }
    send_all(sock, dataout, sizeout);
    
    if ((bytes = recv_all(sock, datain, sizein)) <= 0)
    {
        if (bytes == 0) /* connection closed */
            printf("selectserver: socket %d hung up\n", sock);
        else
            perror("(1) recv");
        
        close(sock); /* bye! */
        bStatus = -1;
    }
    else/* (bytes == nbyte)*/
    {
        debug3("RECEIVING %d,%d\n",sock,sizein);
        int k = 0;
        for (int i=0; i<n; i++)
        {
            for (int j=0; j<sI.dim; j++)
            {
                debug3(" %f (%d)",datain[k],k);
                agents[i]->s->vals[j] = datain[k++];
                if (std::isnan(agents[i]->s->vals[j]) || std::isinf(agents[i]->s->vals[j]))
                    _nan = true;
            }
            
            debug3(" %f (%d)",datain[k],k);
            agents[i]->r = datain[k++];
            if (std::isnan(agents[i]->r) || std::isinf(agents[i]->r))
                _nan = true;
            
            for (int j=0; j<nInfo; j++)
            {
                debug3(" %f (%d)",datain[k],k);
                agents[i]->Info[j] = datain[k++];
            }
            debug3("\n");
            debug2("Got from child %d: reward %f state %s\n", rank, agents[i]->r, agents[i]->s->print().c_str());
            
            if (agents[i]->r < -9.09)
                bStatus = 1;
        }
    }
    if (_nan)
        bStatus = -1;
    return bStatus;
}

int ExternalEnvironment::init()
{
    bStatus = 0;
    bool _nan = false;
    printf("RECEIVING %d,%d\n",sock,sizein);
    if ((bytes = recv_all(sock, datain, sizein)) <= 0)
    {
        if (bytes == 0) /* connection closed */
            printf("selectserver: socket %d hung up before first comm\n", sock);
        else
            perror("(1) recv");
        
        close(sock); /* bye! */
        bStatus = -1;
    }
    else/* (bytes == nbyte)*/
    {
        int k = 0;
        debug3("Got this, asswinkle: ");
        for (int i=0; i<n; i++)
        {
            for (int j=0; j<sI.dim; j++)
            {
                debug3(" %f (%d)",datain[k],k);
                agents[i]->s->vals[j] = datain[k++];
                if (std::isnan(agents[i]->s->vals[j]) || std::isinf(agents[i]->s->vals[j]))
                    _nan = true;
            }
            
            debug3(" %f (%d)",datain[k],k);
            agents[i]->r = datain[k++];
            if (std::isnan(agents[i]->r) || std::isinf(agents[i]->r))
                _nan = true;
            
            for (int j=0; j<nInfo; j++)
            {
                debug3(" %f (%d)",datain[k],k);
                agents[i]->Info[j] = datain[k++];
            }
            debug3("\n");
            
            debug3("Got from child %d: reward %f initial state %s\n", rank, agents[i]->r, agents[i]->s->print().c_str());
            
            if (agents[i]->r < -9.09)
                bStatus = 1;
        }
    }
    if (_nan)
        bStatus = -1;
    return bStatus;
}
void ExternalEnvironment::setDims()
{
    sI.dim = 4;
    
    // State: coordinate...
    sI.bounds.push_back(12);
    sI.top.push_back(2.4);
    sI.bottom.push_back(-2.4);
    sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false); sI.inUse.push_back(true);
    
    // ...velocity...
    sI.bounds.push_back(6);
    sI.top.push_back(1.);
    sI.bottom.push_back(-1.);
    sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false); sI.inUse.push_back(true);
    
    // ...and angular velocity
    sI.bounds.push_back(6);
    sI.top.push_back(1.);
    sI.bottom.push_back(-1.);
    sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false); sI.inUse.push_back(true);
    
    // ...angle...
    sI.bounds.push_back(16);
    sI.top.push_back(0.2);
    sI.bottom.push_back(-0.2);
    sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false); sI.inUse.push_back(true);
    
    //now count the number of states variables and number of actually used
    sI.dim = 0; sI.dimUsed = 0;
    for (int i=0; i<sI.bounds.size(); i++)
    {
        sI.dim++;
        if (sI.inUse[i])
            sI.dimUsed++;
    }
    
    aI.dim = 1;
    
    for (int i=0; i<aI.dim; i++) aI.bounds.push_back(5);
    
    aI.values.push_back(-1.);
    aI.values.push_back(-.1);
    aI.values.push_back(0.0);
    aI.values.push_back(0.1);
    aI.values.push_back(1.0);
    
    nInfo = 0;
    aI.zeroact = 2;
    for (auto& a : agents)
    {
        a->Info.resize(nInfo);
        a->nInfo = nInfo;
        
        //a->setEnvironment(this);
        a->setDims(sI, aI);
        
        a->a = new Action(aI);
        a->s = new State(sI);
    }
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

