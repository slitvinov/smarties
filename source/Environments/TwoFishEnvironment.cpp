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
#include "TwoFishEnvironment.h"


using namespace std;

TwoFishEnvironment::TwoFishEnvironment(vector<Agent*> agents, string execpath, StateType tp, int _rank, const int senses, Settings & settings) :
ExternalEnvironment(agents, execpath, tp, _rank), sight(senses==0 || senses==2), l_line(senses==1 || senses==2), study(settings.rewardType), goalDY(settings.goalDY), gamma(settings.gamma)
{
    if (goalDY > 1.) goalDY = 1. - goalDY; //poor man's sign
#ifndef _scaleR_
    gamma =0; //soory
#endif
}


void TwoFishEnvironment::setDims()
{
    /*
     vector<Real> state(6), info(8+3*NpLatLine);
     state[0] = DX;
     state[1] = dYrel;
     state[2] = ang;
     state[3] = relT;
     state[4] = new_a[ID];
     state[5] = old_a[ID];
     info[0]  = _D[i]->ToD;
     info[1]  = _D[i]->Pout;
     info[2]  = _D[i]->defPower;
     info[3]  = eta;
     info[4]  = _D[i]->EffPDef;
     info[5]  = _D[i]->Vx;
     info[6]  = _D[i]->Vy;
     info[7]  = _D[i]->angVel;
     */
    sI.bounds.clear(); sI.top.clear();
    sI.bottom.clear(); sI.aboveTop.clear();
    sI.belowBottom.clear(); sI.isLabel.clear();
    
    {
        // State: Horizontal distance from goal point...
        sI.bounds.push_back(1); //one block in between the bounds, one more on each side
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(sight);
        
        // ...vertical distance...
        sI.bounds.push_back(1);
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(sight);
        
        // ...inclination of1the fish...
        sI.bounds.push_back(1); // only positive or negative
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(sight);
        
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
        sI.bounds.push_back(1); //T / D 6
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);

        sI.bounds.push_back(1); // Pout 7
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);

        sI.bounds.push_back(1); // defPower 8
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // eta = kin/kin+Pout 9
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // etaPdef 10
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // Vx 11
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // Vy 12
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
        
        sI.bounds.push_back(1); // angvel 13
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
    }
    for (int i=0; i<10; i++) // >=14
    {
        sI.bounds.push_back(1); // (p_above  ) x 10
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
    }
    
    for (int i=0; i<10; i++) // >=24
    {
        sI.bounds.push_back(1); // ( p_below ) x 10
        sI.top.push_back(1.); sI.bottom.push_back(-1.);
        sI.aboveTop.push_back(true); sI.belowBottom.push_back(true);
        sI.isLabel.push_back(false); sI.inUse.push_back(false);
    }
    
    for (int i=0; i<10; i++) // >=34
    {
        sI.bounds.push_back(1); // ( curvature) x 10
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
        
        //a->setEnvironment(this);
        a->setDims(sI, aI);
        
        a->a = new Action(aI);
        a->s = new State(sI);
    }
}

bool TwoFishEnvironment::pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward)
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
    
    Real ToDmax(2.55764), PoutMax(5.85923e-07), dePowerMax(0.273871), etaMax(1), effMax(1);
    Real ToDmin(0.462123), PoutMin(-2.23674e-06), dePowerMin(0.), etaMin(0.253908), effMin(0.428293);
    bool new_sample;
    if (reward<-9.9) new_sample=true;
    
    if (study == 0)
    {
        Real scaled_effic = 2.*(t_sN.vals[10]-effMin)/(effMax-effMin) -1.;
        reward = scaled_effic*(1.-gamma);
    }
    else if (study == 1)
    {
        Real scaled_effic = 2.*(t_sN.vals[10]-effMin)/(effMax-effMin) -1.;
        reward = scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 2)
    {
        reward = 1. - fabs(t_sN.vals[1] - goalDY)/0.25;
        reward*=(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 3)
    {
        Real scaled_effic = 2.*(t_sN.vals[6]-ToDmin)/(ToDmax-ToDmin) -1.;
        reward = scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 4)
    {
        Real scaled_effic = 2.*(t_sN.vals[7]-PoutMin)/(PoutMax-PoutMin) -1.;
        reward = scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 5)
    {
        Real scaled_effic = 2.*(t_sN.vals[8]-dePowerMin)/(dePowerMax-dePowerMin) -1.;
        reward = scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 6)
    {
        Real scaled_effic = 2.*(t_sN.vals[9]-etaMin)/(etaMax-etaMin) -1.;
        reward = scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 7)
    {
        Real scaled_effic = 2.*(t_sN.vals[10]-effMin)/(effMax-effMin) -1.;
        reward = - scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 8)
    {
        Real scaled_effic = 2.*(t_sN.vals[7]-PoutMin)/(PoutMax-PoutMin) -1.;
        reward = - scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 9)
    {
        Real scaled_effic = 2.*(t_sN.vals[8]-dePowerMin)/(dePowerMax-dePowerMin) -1.;
        reward = - scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else if (study == 10)
    {
        Real scaled_effic = 2.*(t_sN.vals[6]-ToDmin)/(ToDmax-ToDmin) -1.;
        reward = - scaled_effic*(1.-gamma);
        if (new_sample) reward = -1.;
    }
    else
    {
        die("Wrong reward\n");
    }
    
    return new_sample;
}

/*
void TwoFishEnvironment::setDims()
{
    sI.dim = 6;
    sI.bounds.clear();
    sI.top.clear();
    sI.bottom.clear();
    sI.aboveTop.clear();
    sI.belowBottom.clear();
    sI.isLabel.clear();
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
    sI.top.push_back(1.);
    sI.bottom.push_back(-1.);
    sI.aboveTop.push_back(true);
    sI.belowBottom.push_back(true);
    sI.isLabel.push_back(false);
    
    // ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
    sI.bounds.push_back(2); // Will get ~ 0 or 0.5
    sI.top.push_back(0.5);
    sI.bottom.push_back(0.0);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(false);
    
    // ...last action (HAX!)
    sI.bounds.push_back(5);
    sI.top.push_back(5.0);
    sI.bottom.push_back(0.0);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(true);
    
    // ...second last action (HAX!)
    sI.bounds.push_back(5);
    sI.top.push_back(5.0);
    sI.bottom.push_back(0.0);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(true);
    
    aI.dim = 1; //How many actions taken per turn by one agent
    
    for (int i=0; i<aI.dim; i++) aI.bounds.push_back(5); //Number of possible actions to choose from (nothing, curve right, curve left)
    
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
    
    nInfo = 38;
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
*/
