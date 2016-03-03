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
#include <iostream>
#include <algorithm>

using namespace std;

#include "oldEnvironment.h"

oldEnvironment::oldEnvironment(vector<Agent*> agents, string execpath, StateType tp, int rank) :
Environment(agents), execpath(execpath)
{
    n = agents.size();
    
    for (auto a : agents)
        exagents.push_back(static_cast<ExternalAgent*>(a));
    
    rewards.resize(n);
    states.resize(n);
    
    sI.type = tp;
}

void oldEnvironment::setup_Comm()
{
    int outpipe[2], inpipe[2];
    
    if(pipe(inpipe))         die("Inpipe error\n");
    
    if(pipe(outpipe))        die("Outpipe error\n");
    
    if( (pid=fork()) == -1 ) die("Couldn't fork!");
    
    if(pid)
    {
        close(inpipe[1]);
        close(outpipe[0]);
        
        fin  = fdopen(inpipe[0],  "r");
        if (!fin)  die("Couldn't create stream for input pipe!");
        
        fout = fdopen(outpipe[1], "w");
        if (!fout) die("Couldn't create stream for output pipe!");
    }
    else
    {
        printf("About to exec.... \n");
        cout << execpath << endl;
        
        //mkdir( ("simulation_"+to_string(rank)+"_"+to_string(index)+"/").c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH );
        //chdir( ("simulation_"+to_string(rank)+"_"+to_string(index)+"/").c_str() );
        
        close(inpipe[0]);
        close(outpipe[1]);
        dup2(inpipe[1], 2);
        dup2(outpipe[0], 0);
        
        freopen("output.txt", "a", stdout);
        
        if (execlp(execpath.c_str(), execpath.c_str(), NULL) == -1)
            die("Unable to exec file '%s'!", execpath.c_str());
    }
}
void oldEnvironment::setDims()
{
    sI.dim = 4;
    
    // State: coordinate...
    sI.bounds.push_back(12);
    sI.top.push_back(2.4);
    sI.bottom.push_back(-2.4);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false);
    
    // ...velocity...
    sI.bounds.push_back(6);
    sI.top.push_back(1.);
    sI.bottom.push_back(-1.);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false);
    
    // ...and angular velocity
    sI.bounds.push_back(6);
    sI.top.push_back(1.);
    sI.bottom.push_back(-1.);
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
    
    nInfo = 0;
    aI.zeroact = 2;
    for (auto& a : exagents)
    {
        a->Info.resize(nInfo);
        a->nInfo = nInfo;
    }
}

int oldEnvironment::evolve(double t)
{
    bStatus = 0;
    
    fprintf(fout, "Actions:\n");
    for (auto& a : exagents)
    {
        fprintf(fout, "%d ", a->a->vals[0]);
        debug2("Sent child: action %d\n", a->a->vals[0]);
    }
    fprintf(fout, "\n");
    fflush(fout);
    
    string sstr;
    do
    {
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
        
        sstr = str;
        sstr.erase(sstr.find_last_not_of(" \t\f\v\n\r")+1);
    }
    while(sstr != "States and rewards:");
    
    
    for (auto& a : exagents)
    {
        double aa, b, c;
        
        for (int i=0; i<a->s->sInfo.dim; i++)
            fscanf(fin, "%lf", &(a->s->vals[i]));
        fscanf(fin, "%lf", &(a->r));
        for (int j=0; j<nInfo; j++)
            fscanf(fin, "%lf", &(a->Info[j]));
        
        debug2("Got from child %d: reward %f,  state %s\n", pid, a->r, a->s->print().c_str());
        if (a->r < -0.99)
            bStatus = 1;
    }
    
    return bStatus;
}

int oldEnvironment::init()
{
    bStatus = 0;
    string sstr;
    do
    {
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
        
        sstr = str;
        sstr.erase(sstr.find_last_not_of(" \t\f\v\n\r")+1);
    }
    while(sstr != "States and rewards:");
    
    
    for (auto& a : exagents)
    {
        double aa, b, c;
        
        for (int i=0; i<a->s->sInfo.dim; i++)
            fscanf(fin, "%lf", &(a->s->vals[i]));
        fscanf(fin, "%lf", &(a->r));
        for (int j=0; j<nInfo; j++)
            fscanf(fin, "%lf", &(a->Info[j]));
        
        debug2("Got from child %d: reward %f ,  state %s\n", pid, a->r, a->s->print().c_str());
        if (a->r < -0.99)
            bStatus = 1;
    }

    return bStatus;
}
