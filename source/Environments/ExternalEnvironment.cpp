/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Dmitry Alexeev on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include <sys/types.h>
#include <unistd.h>

#include "ExternalEnvironment.h"

ExternalEnvironment::ExternalEnvironment(vector<Agent*> agents, string execpath, StateType tp) :
Environment(agents), execpath(execpath)
{
    int outpipe[2], inpipe[2];

    if(pipe(inpipe))
        die("Inpipe error\n");
    if(pipe(outpipe))
        die("Outpipe error\n");

    int pid;
    if( (pid=fork()) == -1 )
        die("Couldn't fork!");

    if(pid)
    {
        close(inpipe[1]);
        close(outpipe[0]);

        fin  = fdopen(inpipe[0],  "r");
        if (!fin)  die("Couldn't create stream for input pipe!");

        fout = fdopen(outpipe[1], "w");
        if (!fout) die("Couldn't create stream for output pipe!");

        int n;
        fscanf(fin, "%d agents", &n);
        if (n != agents.size())
            die("Wrong number of agents!");

        _info("Child simulation started with %d agents\n", n);

        for (auto a : agents)
            exagents.push_back(static_cast<ExternalAgent*>(a));

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
        close(inpipe[0]);
        close(outpipe[1]);
        dup2(inpipe[1], 2);
        dup2(outpipe[0], 0);

        if (execlp(execpath.c_str(), execpath.c_str(), NULL) == -1)
            die("Unable to exec file '%s'!", execpath.c_str());
    }
}

void ExternalEnvironment::setDims()
{
    sI.dim = 4;
    // State: coordinate...
    sI.bounds.push_back(4);
    sI.top.push_back(1);
    sI.bottom.push_back(-1);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);

    // ...velocity...
    sI.bounds.push_back(4);
    sI.top.push_back(1);
    sI.bottom.push_back(-1);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);

    // ...angle...
    sI.bounds.push_back(20);
    sI.top.push_back(0.3);
    sI.bottom.push_back(-0.3);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);

    // ...and angular velocity
    sI.bounds.push_back(4);
    sI.top.push_back(1);
    sI.bottom.push_back(-1);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);

    aI.dim = 1;
    for (int i=0; i<aI.dim; i++) aI.bounds.push_back(5);
}

void ExternalEnvironment::evolve(double t)
{
    fprintf(fout, "Actions:\n");
    for (auto& a : exagents)
        fprintf(fout, "%d ", a->a->vals[0]);
    fprintf(fout, "\n");
    fflush(fout);

    char str[1000];
    fgets(str, 1000, fin);
    fgets(str, 1000, fin);
    if (string(str) != "States and rewards:\n")
        die("Unrecognized command '%s' from child process!\n", str);

    for (auto& a : exagents)
    {
        double aa, b, c;

        fscanf(fin, "%lf %lf %lf %lf %lf", &(a->s->vals[0]), &(a->s->vals[1]), &(a->s->vals[2]), &(a->s->vals[3]), &(a->r));
        debug2("Got from child: reward %f,  state %s\n", a->r, a->s->print().c_str());
    }
}



