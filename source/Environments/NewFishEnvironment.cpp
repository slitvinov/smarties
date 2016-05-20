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
#include "NewFishEnvironment.h"


using namespace std;

NewFishEnvironment::NewFishEnvironment(vector<Agent*> agents, string execpath, StateType tp, int _rank, const int senses, Settings & settings) :
ExternalEnvironment(agents, execpath, tp, _rank), sight(senses==0 || senses==2), l_line(senses==1 || senses==2), study(settings.rewardType), goalDY(settings.goalDY), gamma(settings.gamma)
{
}


void NewFishEnvironment::setDims()
{
    /*
     int k(0);
     for (int j=0; j<NpLatLine; j++) state[20+k++] = _D[i]->VelNAbove[j];
     for (int j=0; j<NpLatLine; j++) state[20+k++] = _D[i]->VelTAbove[j];
     for (int j=0; j<NpLatLine; j++) state[20+k++] = _D[i]->VelNBelow[j];
     for (int j=0; j<NpLatLine; j++) state[20+k++] = _D[i]->VelTBelow[j];
     for (int j=0; j<NpLatLine; j++) state[20+k++] = _D[i]->FPAbove[j];
     for (int j=0; j<NpLatLine; j++) state[20+k++] = _D[i]->FVAbove[j];
     for (int j=0; j<NpLatLine; j++) state[20+k++] = _D[i]->FPBelow[j];
     for (int j=0; j<NpLatLine; j++) state[20+k++] = _D[i]->FVBelow[j];
     */
    
    sI.bounds.clear(); sI.top.clear();
    sI.bottom.clear(); sI.aboveTop.clear();
    sI.belowBottom.clear(); sI.isLabel.clear();
    {
        // State: Horizontal distance from goal point...
        sI.bounds.push_back(1); //one block in between the bounds, one more on each side
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        // ...vertical distance...
        sI.bounds.push_back(1);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        // ...inclination of1the fish...
        sI.bounds.push_back(1); // only positive or negative
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        // ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
        sI.bounds.push_back(1); // Will get ~ 0 or 0.5
        sI.top.push_back(0.5); sI.bottom.push_back(0.0);
        sI.aboveTop.push_back(false); sI.belowBottom.push_back(false);
        sI.isLabel.push_back(false); sI.inUse.push_back(true);
        // ...last action (HAX!)
        sI.bounds.push_back(1);
        sI.top.push_back(5.0); sI.bottom.push_back(0.0);
        sI.aboveTop.push_back(false); sI.belowBottom.push_back(false);
        sI.isLabel.push_back(true); sI.inUse.push_back(true);
        // ...second last action (HAX!)
        sI.bounds.push_back(1);
        sI.top.push_back(5.0); sI.bottom.push_back(0.0);
        sI.aboveTop.push_back(false); sI.belowBottom.push_back(false);
        sI.isLabel.push_back(true); sI.inUse.push_back(true); //if l_line i have curvature info
    }
    {
        sI.bounds.push_back(1); //Dist 6
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);

        sI.bounds.push_back(1); //Quad 7
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);

        sI.bounds.push_back(1); // VxAvg 8
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // VyAvg 9
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // AvAvg 10
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
    }
    {
        sI.bounds.push_back(1); //Pout 11
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); //defPower 12
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // EffPDef 13
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // PoutBnd 14
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // defPowerBnd 15
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // EffPDefBnd 16
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // Pthrust 17
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // Pdrag 18
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // ToD 19
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
    }

    for (int i=0; i<20; i++)
    {
        sI.bounds.push_back(1); // (Vel  ) x 20
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
    }
    for (int i=0; i<20; i++)
    {
        sI.bounds.push_back(1); // (Force ) x 20
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
    }
    
    //now count the number of states variables and number of actually used
    sI.dim = 0; sI.dimUsed = 0;
    for (int i=0; i<sI.bounds.size(); i++)
    {
        sI.dim++;
        if (sI.inUse[i])
            sI.dimUsed++;
    }
    
    aI.dim = 1; //How many actions taken per turn by one agent MUST BE 1 (TODO?!??!!)
    //for (int i=0; i<aI.dim; i++)
        aI.bounds.push_back(5); //Number of possible actions to choose from
    
    aI.values.push_back(-2.);
    aI.values.push_back(-1.);
    aI.values.push_back(0.0);
    aI.values.push_back(1.0);
    aI.values.push_back(2.0);
    
    sI.values.push_back(-2.);
    sI.values.push_back(-1.);
    sI.values.push_back(0.0);
    sI.values.push_back(1.0);
    sI.values.push_back(2.0);
    
    nInfo = 0;
    aI.zeroact = 2;
    for (auto& a : agents)
    {
        a->Info.resize(nInfo);
        a->nInfo = nInfo;
        a->setDims(sI, aI);
        a->a = new Action(aI);
        a->s = new State(sI);
    }
}

bool NewFishEnvironment::pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward)
{
    /*
     if (fabs(t_sN.vals[4] - t_a.vals[0])>0.001)
     {
     printf("ASSUMING 2F: mismatch between state and reported action!!! %s \n",t_sN.printClean().c_str());
     abort();
     }
     if (fabs(t_sN.vals[5] - t_sO.vals[4])>0.001)
     {
     printf("ASSUMING 2F: mismatch between new state and old state!!! %s \n",t_sN.printClean().c_str());
     abort();
     }
     if ( fabs(t_sN.vals[3] - t_sO.vals[3])<1e-2 )
     {
     printf("ASSUMING 2F: same time for two states!!! %s \n",t_sN.printClean().c_str());
     abort();
     }
     */
    for (int i(0); i<20; i++)
    {
        max_scale[i] = std::max(max_scale[i], t_sN.vals[i]);
        min_scale[i] = std::min(min_scale[i], t_sN.vals[i]);
    }

    bool new_sample(false);
    if (reward<-9.9) new_sample=true;
    
         if (study == 0)
    {
#ifndef _scaleR_
        reward = 2*t_sN.vals[13]-1;
#else
        reward = (2*(t_sN.vals[13]-0.4)/(1.-.04) -1.)*(1.-gamma);
#endif
        if (new_sample) reward = -1.;
    }
    else if (study == 1)
    {
        reward = (2*t_sN.vals[16]-1.);//*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 2)
    {
        Real scaled_effic = 1. - fabs(t_sN.vals[1] - goalDY)/0.5;
        reward = scaled_effic;//*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else
    {
        die("Wrong reward\n");
    }
    
    return new_sample;
}
