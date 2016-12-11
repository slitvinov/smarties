/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "Environment.h"
//#include <sys/types.h>
//#include <sys/time.h>
#include <sys/stat.h>
//#include <cstdio>
//#include <unistd.h>
//#include <errno.h>
//#include <math.h>
#include <signal.h>
//#include <iostream>
//#include <algorithm>
//#include <stdio.h>

#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>

using namespace std;


static void parse(char *line, char **argv)
{
    while (*line != '\0') {		/* if not the end of line ....... */
        while (*line == ' ' || *line == '\t' || *line == '\n')
            *line++ = '\0';		/* replace white spaces with 0 */
        *argv++ = line;		/* save the argument position */
        while (*line != '\0' && *line != ' ' &&
               *line != '\t' && *line != '\n')
            line++;	/* skip the argument until ...*/
    }
    *argv = '\0';	/* mark the end of argument list */
}

static int recv_all(int fd, void *buffer, unsigned int size)
{
    int result;
    unsigned int s=size;
    char *pos = (char*)buffer;


    do {
        result=recv(fd,pos,s,0);
        if((result!=-1)&&(result>0)) {
            s -= result;
            pos += result;
        }
        else
            return result; /*-1;*/
    } while (s>0);
    return size;
}

static int send_all(int fd, void *buffer, unsigned int size)
{
    int result;
    unsigned int s=size;
    char *pos = (char*)buffer;

    do {
        result=send(fd,pos,s,0);
        if((result!=-1)&&(result>0)) {
            s -= result;
            pos += result;
        }
        else return result; /*-1;*/
    } while (s>0);

    //printf("sender %f\n",*((double*)buffer));
    return size;
}

Environment::Environment(const int _nAgents, const string _execpath, const int _rank, Settings & settings) :
execpath(_execpath), rank(_rank), g(settings.gen), nAgents(_nAgents), resetAll(true), workerid(_rank),
sock(0), ListenerSocket(0), bytes(0), iter(0), max_scale(20, -1000), min_scale(20, 1000), gamma(settings.gamma)
{
    for (int i=0; i<nAgents; i++) agents.push_back(new Agent(i));
}

void Environment::setup_Comm()
{
    workerid = rank;
    string dummy = "/tmp/smarties_sock_";
    sprintf(SOCK_PATH, "%s%d", dummy.c_str(), workerid);
    printf("mserver: SOCK_PATH=->%s<-\n", SOCK_PATH);

    sizein = (2 + sI.dim + 1)*sizeof(double); //nagent, tag, state, reward
    datain = (double *) malloc(sizein);
    sizeout = aI.dim * sizeof(double);
    dataout = (double *) malloc(sizeout);

    spawn_server();
    //printf("comm dim = %d %d \n", sizein, sizeout);
    sock = socket(AF_UNIX, SOCK_STREAM, 0);

    int _true = 1;
    if(setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &_true, sizeof(int))<0) {
	     die("Sockopt failed\n");
    }

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

Environment::~Environment()
{
    close(sock);
    _dispose_object(datain);
    _dispose_object(dataout);
    for (auto & trash : agents)
      _dispose_object( trash);
}

void Environment::close_Comm()
{
    close(ListenerSocket);
    free(datain);
    free(dataout);
}

void Environment::spawn_server()
{
    const int rf = fork();
    if (rf == 0) {
        char line[1024];
        char *largv[64];

        mkdir(("simulation_"+to_string(rank)+"_"+to_string(iter)+"/").c_str(),
                                        S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        chdir(("simulation_"+to_string(rank)+"_"+to_string(iter)+"/").c_str());

        sprintf(line, execpath.c_str());
        parse(line, largv);     // prepare argv

        #if 1==1 //if true goes to stdout
        char output[256];
        sprintf(output, "output");
        int fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        dup2(fd, 1);    // make stdout go to file
        dup2(fd, 2);    // make stderr go to file
        close(fd);      // fd no longer needed
        #endif

        printf("About to exec.... \n");
        cout << execpath << endl << *largv << endl;

        //int res = execlp(execpath.c_str(), execpath.c_str(), NULL);
        const int res = execlp(execpath.c_str(),
                               execpath.c_str(),
                               to_string(workerid).c_str(),
                               NULL);
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
        dataout[i] = (double) agents[iAgent]->a->vals[i];
        assert(not std::isnan(agents[iAgent]->a->vals[i]) &&
        	   not std::isinf(agents[iAgent]->a->vals[i]));
    }
    //debug3("Sent child %d: action %s\n", rank, agents[iAgent]->a->print().c_str()); fflush(0);
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
        assert(iAgent<nAgents);
        bStatus = *((int*) (datain+1)); //first (==1?), terminal (==2?), etc
        //debug3("Receiving from agent %d %d: ", iAgent, bStatus);

        std::swap(agents[iAgent]->s,agents[iAgent]->sOld);

        int k = 2;
        for (int j=0; j<sI.dim; j++) {
            //debug3(" %f (%d)",datain[k],k);
            agents[iAgent]->s->vals[j] = (Real) datain[k++];
            assert(not std::isnan(agents[iAgent]->s->vals[j]) &&
                   not std::isinf(agents[iAgent]->s->vals[j]));
        }

        //debug3(" %f (%d)\n",datain[k],k);
        agents[iAgent]->r = (Real) datain[k++];
        assert(not std::isnan(agents[iAgent]->r) &&
               not std::isinf(agents[iAgent]->r));
        //debug3("Got from child %d: reward %f state %s\n",
        //rank, agents[iAgent]->r, agents[iAgent]->s->print().c_str());
        fflush(0);
    }
    fflush(0);
    return bStatus;
}

bool Environment::predefinedNetwork(Network* const net) const
{
	//this function can be used if environment requires particular network settings
	//i.e. not fully connected LSTM/FF network
	//i.e. if you want to use convolutions
	return false;
}

void Environment::commonSetup()
{
    sI.dim = 0; sI.dimUsed = 0;
    for (int i=0; i<sI.inUse.size(); i++) {
        sI.dim++;
        if (sI.inUse[i]) sI.dimUsed++;
    }
    printf("State has %d component, %d in use\n", sI.dim, sI.dimUsed);

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
