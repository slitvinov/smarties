/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
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
#include "HardCartEnvironment.h"


using namespace std;

HardCartEnvironment::HardCartEnvironment(vector<Agent*> agents, string execpath, StateType tp, int _rank) :
ExternalEnvironment(agents, execpath, tp, _rank)
{ }

void HardCartEnvironment::setDims()
{
    
    sI.dim = 2;
    // State: coordinate...
    sI.bounds.push_back(12);
    sI.top.push_back(2.0);
    sI.bottom.push_back(-2.0);
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
    
    aI.values.push_back(-2.);
    aI.values.push_back(-.5);
    aI.values.push_back(0.0);
    aI.values.push_back(0.5);
    aI.values.push_back(2.0);
    
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
