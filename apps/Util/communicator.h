#pragma once
#include "util.h"
#include <sys/stat.h>
#include <iostream>
#include <cstdio>
#include <stdio.h>
#include <unistd.h>
//#include <omp.h>
#include <vector>
#include <fstream>

class Communicator
{
    int workerid, nAgents, probdim, sock, sizein, sizeout, servlen;
    char SOCK_PATH[256];
    struct sockaddr_un server_addr;
    int k, a;
public:
    double *datain, *dataout;
    
    void sendState(std::vector<double> & state, double reward)
    {
        //std::ofstream fout;
        //fout.open("log.dat",std::ios::app);
        
        for (int j=0; j<state.size(); j++)
        {
            //fout << "Sent state "<<k<<"\n";
            //printf("Sent state %d %f\n",k, state[j]);
            *(dataout +k++) = state[j];
        }
        
        //fout << "Sent state "<<k<<"\n";
        //printf("Sent state %d %f\n",k, reward);
        
        *(dataout +k++) = reward;
        if (k==probdim)
        {
            k = 0; a = 0;
            //double t1 = omp_get_wtime();

            send_all(sock, dataout, sizeout);
            
            if (reward>-.99)
            recv_all(sock, datain, sizein);
            
            //for (int i=0; i<nAgents; i++)  printf("comm %f\n", *(datain +i));
            //double t2 = omp_get_wtime();
            //printf("CLIENT: Elapsed time %.3lf us \n", 1e6*(t2-t1));
        }
        else if (k>probdim)
        {
            perror("You messed up the size of your state space/ nagents\n");
            //fout << "You messed up the size of your state space/ nagents "<<k<<"\n";
            abort();
        }
        //fout.close();
    }
    
    int recvAction()
    {
        //std::ofstream fout;
        //fout.open("log.dat",std::ios::app);
        //printf("comm %f\n", *(datain +a));
        int act = (int) *(datain +a++);
        if (a>probdim)
        {
            //fout << "You messed up the size of your state space/ nagents "<<k<<"\n";
            perror("You messed up the size of your action space/ nagents\n");
            abort();
        }
        return act;
        
        //fout.close();
    }
    
    ~Communicator()
    {
        close(sock);
    }
    
    Communicator(int _probdim, int _nAgents): workerid(0), nAgents(_nAgents), probdim(_probdim), sock(0), sizein(0), k(0), a(0)
    {
#if 0
        char output[256];
        sprintf(output, "output_%d", workerid);
        int fd = open(output, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        
        dup2(fd, 1);    // make stdout go to file
        dup2(fd, 2);    // make stderr go to file
        close(fd);      // fd no longer needed
#endif
        sprintf(SOCK_PATH, "%s%d", "/tmp/sock_", workerid);
        printf("SOCK_PATH=->%s<-\n", SOCK_PATH);
        
        probdim = (probdim+1)*nAgents;
        sizeout = probdim*sizeof(double);
        dataout = (double *) malloc(probdim);
        sizein  = nAgents*sizeof(double);
        datain  = (double *) malloc(nAgents);
        /* Create a socket */
        printf("recv problem dim = %d %d %d %d \n", probdim, nAgents, sizein, sizeout);
        sock = socket(AF_UNIX, SOCK_STREAM, 0);
        
        /* Specify the server */
        bzero((char *)&server_addr, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strcpy(server_addr.sun_path, SOCK_PATH);
        servlen = sizeof(server_addr.sun_family) + strlen(server_addr.sun_path);
        
        /* Connect to the server */
        while (connect(sock, (struct sockaddr *)&server_addr, servlen) < 0)
            perror("connecting...\n");
            //exit(1);	// here, we can sleep and retry
    }
    
    void dbg(double *x, int *pn)
    {
        int i, n = *pn;
        int me = getpid();	/* spanwer_id : workerid */
        
        printf("spanwer(%d): running task with params (", me);
        for (i = 0; i < n-1; i++)
            printf("%.6lf,", x[i]);
        printf("%.6lf)\n", x[i]);
        fflush(0);
    }
};