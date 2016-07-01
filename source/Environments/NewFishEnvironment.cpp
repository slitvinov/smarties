/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "NewFishEnvironment.h"

using namespace std;

NewFishEnvironment::NewFishEnvironment(const int nAgents, const string execpath, const int _rank, Settings & settings) :
Environment(nAgents, execpath, _rank, settings), sight(settings.senses==0), POV(settings.senses==1),
l_line(settings.senses==2), p_sensors(settings.senses==3), study(settings.rewardType), gamma(settings.gamma),
goalDY((settings.goalDY>1.)? 1.-settings.goalDY : settings.goalDY)
{
}

void NewFishEnvironment::setDims()
{
    {
        sI.bounds.clear(); sI.top.clear(); sI.bottom.clear(); sI.isLabel.clear(); sI.inUse.clear();
        {
            // State: Horizontal distance from goal point...
            sI.bounds.push_back(1); //one block in between the bounds, one more on each side
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(sight);
            // ...vertical distance...
            sI.bounds.push_back(1);
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(sight);
            // ...inclination of1the fish...
            sI.bounds.push_back(1); // only positive or negative
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(sight || POV);
            // ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
            sI.bounds.push_back(1); // Will get ~ 0 or 0.5
            sI.top.push_back(.5); sI.bottom.push_back(0.0);
            sI.isLabel.push_back(false); sI.inUse.push_back(true);
            // ...last action (HAX!)
            sI.bounds.push_back(1);
            sI.top.push_back(.5); sI.bottom.push_back(-.5);
            sI.isLabel.push_back(false); sI.inUse.push_back(true);
            // ...second last action (HAX!)
            sI.bounds.push_back(1);
            sI.top.push_back(.5); sI.bottom.push_back(-.5);
            sI.isLabel.push_back(false); sI.inUse.push_back(true); //if l_line i have curvature info
        }
        {
            sI.bounds.push_back(1); //Dist 6
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(POV);

            sI.bounds.push_back(1); //Quad 7
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(POV);

            sI.bounds.push_back(1); // VxAvg 8
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(p_sensors || l_line);
            
            sI.bounds.push_back(1); // VyAvg 9
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(p_sensors || l_line);
            
            sI.bounds.push_back(1); // AvAvg 10
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(p_sensors || l_line);
        }
        {
            sI.bounds.push_back(1); //Pout 11
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
            
            sI.bounds.push_back(1); //defPower 12
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
            
            sI.bounds.push_back(1); // EffPDef 13
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
            
            sI.bounds.push_back(1); // PoutBnd 14
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
            
            sI.bounds.push_back(1); // defPowerBnd 15
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
            
            sI.bounds.push_back(1); // EffPDefBnd 16
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
            
            sI.bounds.push_back(1); // Pthrust 17
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
            
            sI.bounds.push_back(1); // Pdrag 18
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
            
            sI.bounds.push_back(1); // ToD 19
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(false);
        }

        for (int i=0; i<5; i++) {
            sI.bounds.push_back(1); // (VelNAbove  ) x 5 [20]
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(l_line);
        }
        for (int i=0; i<5; i++) {
            sI.bounds.push_back(1); // (VelTAbove  ) x 5 [25]
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(l_line);
        }
        for (int i=0; i<5; i++) {
            sI.bounds.push_back(1); // (VelNBelow  ) x 5 [30]
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(l_line);
        }
        for (int i=0; i<5; i++) {
            sI.bounds.push_back(1); // (VelTBelow  ) x 5 [35]
            sI.top.push_back(1.); sI.bottom.push_back(-1.);
            sI.isLabel.push_back(false); sI.inUse.push_back(l_line);
        }
        for (int i=0; i<5; i++) {
            sI.bounds.push_back(1); // (FPAbove  ) x 5 [40]
            sI.top.push_back(0.1); sI.bottom.push_back(-0.1);
            sI.isLabel.push_back(false); sI.inUse.push_back(p_sensors);
        }
        for (int i=0; i<5; i++) {
            sI.bounds.push_back(1); // (FVAbove  ) x 5 [45]
            sI.top.push_back(1e-4); sI.bottom.push_back(-1e-4);
            sI.isLabel.push_back(false); sI.inUse.push_back(p_sensors);
        }
        for (int i=0; i<5; i++) {
            sI.bounds.push_back(1); // (FPBelow  ) x 5 [50]
            sI.top.push_back(0.1); sI.bottom.push_back(-0.1);
            sI.isLabel.push_back(false); sI.inUse.push_back(p_sensors);
        }
        for (int i=0; i<5; i++) {
            sI.bounds.push_back(1); // (FVBelow ) x 5 [55]
            sI.top.push_back(1e-4); sI.bottom.push_back(-1e-4);
            sI.isLabel.push_back(false); sI.inUse.push_back(p_sensors);
        }
        /*
        sI.values.push_back(-.50);
        sI.values.push_back(-.25);
        sI.values.push_back(0.00);
        sI.values.push_back(0.25);
        sI.values.push_back(0.50);
         */
    }
    {
        aI.realValues = false;
        aI.dim = 1;
        aI.zeroact = 2;
        aI.values.resize(aI.dim);
        
        for (int i=0; i<aI.dim; i++) {
            aI.bounds.push_back(5); //Number of possible actions to choose from
            aI.upperBounds.push_back(0.5);
            aI.lowerBounds.push_back(-.5);
            
            aI.values[i].push_back(-.50);
            aI.values[i].push_back(-.25);
            aI.values[i].push_back(0.00);
            aI.values[i].push_back(0.25);
            aI.values[i].push_back(0.50);
        }
    }
    resetAll=true;
    commonSetup();
}

bool NewFishEnvironment::pickReward(const State & t_sO, const Action & t_a,
                                    const State & t_sN, Real & reward)
{/*
    if (fabs(t_sN.vals[5] -t_sO.vals[4])>0.001) {
        printf("Mismatch new and old state!!! %s\n",t_sN.print().c_str());
        abort();
    }
    if (fabs(t_sN.vals[4] -t_a.valsContinuous[0])>0.001) {
        printf("Mismatch state and action!!! %s\n",t_sN.print().c_str());
        abort();
    }
    if ( fabs(t_sN.vals[3] -t_sO.vals[3])<1e-2 ) {
        printf("Same time for two states!!! %s\n",t_sN.print().c_str());
        abort();
    }
    if (t_sN.vals[13]>1 || t_sN.vals[13]<0) {
        printf("You modified the efficiency\n");
        abort();
    }
    if (t_a.vals[0]<0 || t_a.vals[0]>4) {
        printf("Actions out of bounds\n");
        abort();
    }
    if (fabs(t_a.valsContinuous[0]-aI.values[0][round(t_a.vals[0])])>1e-2) {
        printf("You modified the actions\n");
        abort();
    }*/
    
    for (int i(0); i<20; i++) {
        max_scale[i] = std::max(max_scale[i], t_sN.vals[i]);
        min_scale[i] = std::min(min_scale[i], t_sN.vals[i]);
    }

    bool new_sample(false);
    if (reward<-9.9) new_sample=true;
    
    if (study == 0) {
#ifndef _scaleR_
        reward = 2*t_sN.vals[13]-1;
#else
        reward = (1.-gamma)*(t_sN.vals[13]-.4)/(1.-.4);
#endif
        if (new_sample) reward = -1.;
    }
    else if (study == 1) {
#ifndef _scaleR_
        reward = 2*t_sN.vals[16]-1;
#else
        reward = (1.-gamma)*(t_sN.vals[16]-.3)/(1.-.3);
#endif
        if (new_sample) reward = -1.;
    }
    else if (study == 2) {
#ifndef _scaleR_
        reward =  1.-fabs(t_sN.vals[1]-goalDY)/.5;
#else
        reward = (1.-gamma)*(1.-fabs(t_sN.vals[1]-goalDY));
        if (new_sample) reward = -1.;
#endif
    }
    else {
        die("Wrong reward\n");
    }
    
    return new_sample;
}
