/*
 *  CartEnvironment.cpp
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
#include <iostream>
#include <algorithm>

using namespace std;

#include "CartEnvironment.h"

CartEnvironment::CartEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank, int index) :
Environment(agents), execpath(execpath)
{
    int outpipe[2], inpipe[2];
    
    if(pipe(inpipe))
    {
        die("Inpipe error\n");
    }
    if(pipe(outpipe))
    {
        die("Outpipe error\n");
    }
    
    int pid;
    if( (pid=fork()) == -1 )
    {
        die("Couldn't fork!");
    }
    
    if(pid)
    {
        close(inpipe[1]);
        close(outpipe[0]);
        
        fin  = fdopen(inpipe[0],  "r");
        if (!fin)  die("Couldn't create stream for input pipe!");
        
        fout = fdopen(outpipe[1], "w");
        if (!fout) die("Couldn't create stream for output pipe!");
        
        int n;
        if (rank != 0)
        {
            fscanf(fin, "%d agents", &n);
            if (n != agents.size())
            {
                die("Slave %d has a wrong number of agents: n=%d, size=%d!",rank, n, agents.size());
            }
        }
        else
        {
            n = agents.size();
        }
        
        _info("Child simulation started with %d agents\n", n);
        
        for (auto a : agents)
        {
            exagents.push_back(static_cast<CartAgent*>(a));
        }
        
        rewards.resize(n);
        states.resize(n);
        
        sI.type = tp;
        setDims();
        for (auto& a : exagents)
        {
            a->setEnvironment(this);
            a->setDims(sI, aI);
            
            a->a = new Action(aI);
            a->s = new State(sI);
        }
    }
    else
    {
        if (rank == 0) exit(0);
        
        mkdir( ("simulation_"+to_string(rank)+"_"+to_string(index)+"/").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
        chdir( ("simulation_"+to_string(rank)+"_"+to_string(index)+"/").c_str() );
        
        close(inpipe[0]);
        close(outpipe[1]);
        dup2(inpipe[1], 2);
        dup2(outpipe[0], 0);
        
        freopen("output.txt", "a", stdout);
        
        if (execlp(execpath.c_str(), execpath.c_str(), NULL) == -1)
            die("Unable to exec file '%s'!", execpath.c_str());
    }
}

void CartEnvironment::setDims()
{
    sI.dim = 4;
    // State: coordinate...
    sI.bounds.push_back(42);
    sI.top.push_back(2.0);
    sI.bottom.push_back(-2.0);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    
    // ...velocity...
    sI.bounds.push_back(12);
    sI.top.push_back(0.5);
    sI.bottom.push_back(-0.5);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    
    // ...and angular velocity
    sI.bounds.push_back(12);
    sI.top.push_back(0.5);
    sI.bottom.push_back(-0.5);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    
    // ...angle...
    sI.bounds.push_back(22);
    sI.top.push_back(0.2);
    sI.bottom.push_back(-0.2);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    
    aI.dim = 1;
    
    for (int i=0; i<aI.dim; i++) aI.bounds.push_back(5);
}

int CartEnvironment::evolve(double t)
{
    fprintf(fout, "Actions:\n");
    for (auto& a : exagents)
    {
        fprintf(fout, "%d ", a->a->vals[0]);
        debug6("Sent child: action %d\n", a->a->vals[0]);
    }
    fprintf(fout, "\n");
    fflush(fout);
    bRestart = false;
    char str[1000] = "";
    bool empty = true;
    
    while (empty)
    {
        fgets(str, 1000, fin);
        string sstr(str);
        
        sstr.erase(std::remove(sstr.begin(), sstr.end(), '\n'), sstr.end());
        sstr.erase(std::remove(sstr.begin(), sstr.end(), ' '), sstr.end());
        sstr.erase(std::remove(sstr.begin(), sstr.end(), '\t'), sstr.end());
        empty = sstr.length() < 1;
    }
    
    string sstr(str);
    sstr.erase(sstr.find_last_not_of(" \t\f\v\n\r")+1);
    
    if (sstr != "States and rewards:")
    {
        if (sstr != "Restart!") {
            die("Unrecognized command '%s' from child process!\n", str);
        }
        else
        {
            die("Unrecognized command '%s' from child process!\n", str);
            std:cout << "Ending environment" << endl;
            fprintf(fout, "Die\n");
            bRestart = true;
            
        }
    }
    
    for (auto& a : exagents)
    {
        double aa, b, c;
        
        for (int i=0; i<a->s->sInfo.dim; i++)
            fscanf(fin, "%lf", &(a->s->vals[i]));
        fscanf(fin, "%lf", &(a->r));
        
        if (a->r < -99.0)
            bRestart = true;
        debug6("Got from child: reward %f,  state %s\n", a->r, a->s->print().c_str());
    }
    if (bRestart)
        return 1;
    
    return 0;
}
