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

TwoFishEnvironment::TwoFishEnvironment(vector<Agent*> agents, string execpath, StateType tp, int _rank) :
ExternalEnvironment(agents, execpath, tp, _rank)
{ }

void TwoFishEnvironment::setDims()
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
    
    nInfo = 0;
    aI.zeroact = 2;
    for (auto& a : exagents)
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
