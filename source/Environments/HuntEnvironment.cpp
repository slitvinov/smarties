/*
 *  HuntEnvironment.cpp
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

#include "HuntEnvironment.h"


void HuntEnvironment::setDims()
{
    sI.dim = 7;
    
    sI.bounds.push_back(2);
    sI.top.push_back(1.1);
    sI.bottom.push_back(0.0);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(false);
    
    // ...distance...
    sI.bounds.push_back(4);
    sI.top.push_back(4.);
    sI.bottom.push_back(0);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(false);
    
    // ...relative inclination...
    sI.bounds.push_back(5);
    sI.top.push_back(3.1416);
    sI.bottom.push_back(-3.1416);
    sI.aboveTop.push_back(false);
    sI.belowBottom.push_back(false);
    sI.isLabel.push_back(false);
    
    // ...quadrant...
    sI.bounds.push_back(8);
    sI.top.push_back(1.5708);
    sI.bottom.push_back(-1.5708);
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
    
    aI.zeroact = 0;
    for (auto& a : exagents)
    {
        a->Info.resize(2);
        a->nInfo = 2;
    }
}
