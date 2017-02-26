/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "TwoActFishEnvironment.h"

using namespace std;

TwoActFishEnvironment::TwoActFishEnvironment(const int _nAgents,
  const string _execpath, const int _rank, Settings & settings) :
Environment(_nAgents, _execpath, _rank, settings),
sight(settings.senses==0 || settings.senses==4), POV(settings.senses==1),
l_line(settings.senses==2), p_sensors(settings.senses==3 || settings.senses==4),
study(settings.rewardType),
goalDY((settings.goalDY>1.)? 1.-settings.goalDY : settings.goalDY)
{
}

void TwoActFishEnvironment::setDims()
{
    {
        sI.inUse.clear();
        {
            // State: Horizontal distance from goal point...
            sI.inUse.push_back(sight);
            // ...vertical distance...
            sI.inUse.push_back(sight);
            // ...inclination of1the fish...
            sI.inUse.push_back(sight);
            // ..time % Tperiod (phase of the motion, maybe also some info on what is the incoming vortex?)...
            sI.inUse.push_back(true);
            // ...last action (HAX!)
            sI.inUse.push_back(true);
            // ...second last action (HAX!)
            sI.inUse.push_back(false); //if l_line i have curvature info
        }
        {
            //New T period
            sI.inUse.push_back(true);

            //Phase Shift
            sI.inUse.push_back(true);

            // VxInst
            sI.inUse.push_back(true);

            // VyInst
            sI.inUse.push_back(true);

            // AvInst
            sI.inUse.push_back(false);
        }
#if 0
            //Xabs 6
            sI.inUse.push_back(false);

            //Yabs 7
            sI.inUse.push_back(false);
#endif
        {
            //Dist 6
            sI.inUse.push_back(false);

            //Quad 7
            sI.inUse.push_back(false);

            // VxAvg 8
            sI.inUse.push_back(true);

            // VyAvg 9
            sI.inUse.push_back(true);

            // AvAvg 10
            sI.inUse.push_back(true);
        }
        {
            //Pout 11
            sI.inUse.push_back(false);

            //defPower 12
            sI.inUse.push_back(false);

            // EffPDef 13
            sI.inUse.push_back(false);

            // PoutBnd 14
            sI.inUse.push_back(false);

            // defPowerBnd 15
            sI.inUse.push_back(false);

            // EffPDefBnd 16
            sI.inUse.push_back(false);

            // Pthrust 17
            sI.inUse.push_back(false);

            // Pdrag 18
            sI.inUse.push_back(false);

            // ToD 19
            sI.inUse.push_back(false);
        }
        const int nSensors = 20;
        for (int i=0; i<nSensors; i++) {
            // (VelNAbove  ) x 5 [20]
            sI.inUse.push_back(l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (VelTAbove  ) x 5 [25]
            sI.inUse.push_back(l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (VelNBelow  ) x 5 [30]
            sI.inUse.push_back(l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (VelTBelow  ) x 5 [35]
            sI.inUse.push_back(l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (FPAbove  ) x 5 [40]
            sI.inUse.push_back(p_sensors || l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (FVAbove  ) x 5 [45]
            sI.inUse.push_back(p_sensors || l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (FPBelow  ) x 5 [50]
            sI.inUse.push_back(p_sensors || l_line);
        }
        for (int i=0; i<nSensors; i++) {
            // (FVBelow ) x 5 [55]
            sI.inUse.push_back(p_sensors || l_line);
        }
        for (int i=0; i<2*nSensors; i++) {
            // (FVBelow ) x 5 [55]
            sI.inUse.push_back(p_sensors || l_line);
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
        aI.dim = 2;
        aI.bounded = true;
        aI.values.resize(aI.dim);
        //curavture
        aI.bounds.push_back(7); //Number of possible actions to choose from
        aI.values[0].push_back(-.75);
        aI.values[0].push_back(-.50);
        aI.values[0].push_back(-.25);
        aI.values[0].push_back(0.00);
        aI.values[0].push_back(0.25);
        aI.values[0].push_back(0.50);
        aI.values[0].push_back(0.75);
        //period:
        aI.bounds.push_back(7); //Number of possible actions to choose from
        aI.values[1].push_back(-.5);
        aI.values[1].push_back(-.25);
        aI.values[1].push_back(-.125);
        aI.values[1].push_back(0.00);
        aI.values[1].push_back(0.125);
        aI.values[1].push_back(0.250);
        aI.values[1].push_back(0.5);
    }
    resetAll=false;
    commonSetup();
}

void TwoActFishEnvironment::setAction(const int & iAgent)
{

    if (agents[iAgent]->a->vals[0] >0.75 ) {
    	printf("Act0 is too large (>0), reassigned to prevent crash\n");
      agents[iAgent]->a->vals[0] = 0.75;
    }
    if (agents[iAgent]->a->vals[0] <-.75 ) {
    	printf("Act0 is too large (<0), reassigned to prevent crash\n");
      agents[iAgent]->a->vals[0] = -.75;
    }
    if (agents[iAgent]->a->vals[1] >0.5 ) {
    	printf("Act1 is too large (>0), reassigned to prevent crash\n");
      agents[iAgent]->a->vals[1] = 0.5;
    }
    if (agents[iAgent]->a->vals[1] <-.5 ) {
    	printf("Act1 is too large (<0), reassigned to prevent crash\n");
      agents[iAgent]->a->vals[1] = -.5;
    }

    Environment::setAction(iAgent);
}

int TwoActFishEnvironment::getState(int & iAgent)
{
    int bStatus = Environment::getState(iAgent);
    if(std::fabs(agents[iAgent]->a->vals[0])>0.74)
      agents[iAgent]->r = std::min((Real)-1.,agents[iAgent]->r); //gently push sim away from extreme curvature: not kosher
    if(std::fabs(agents[iAgent]->a->vals[1])>0.49)
      agents[iAgent]->r = std::min((Real)-1.,agents[iAgent]->r); //gently push sim away from extreme acceleration: not kosher

    return bStatus;
}

bool TwoActFishEnvironment::pickReward(const State& t_sO, const Action& t_a,
                                const State& t_sN, Real& reward, const int info)
{
    if (fabs(t_sN.vals[4] -t_a.vals[0])>0.001) {
        printf("Mismatch state and action!!! %s === %s\n",
         t_sN.print().c_str(),t_a.print().c_str());
        abort();
    }
    if (fabs(t_sN.vals[6] -t_a.vals[1])>0.001) {
        printf("Mismatch state and action!!! %s === %s\n",
         t_sN.print().c_str(),t_a.print().c_str());
        abort();
    }
    /*
    if ( fabs(t_sN.vals[3] -t_sO.vals[3])<1e-3 ) {
        printf("Same time for two states!!! %s === %s\n",t_sO.print().c_str(),t_sN.print().c_str());
        abort();
    }
    */
    bool new_sample(false);
    if (reward<-9.9) new_sample=true;
    if(new_sample) assert(info==2);

    if (study == 0) {
        const Real scaledEfficiency = (t_sN.vals[18]-.4)/(1.-.4);
        reward = scaledEfficiency;
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
    }
    else if (study == 1) {
        const Real scaledEfficiency = (t_sN.vals[21]-.3)/(.6-.3);
        reward = scaledEfficiency;
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
    }
    else if (study == 2) {
        reward =  1.-fabs(t_sN.vals[1]-goalDY)/.5;
    }
    else if (new_sample) reward = -10.;

    if(std::fabs(t_a.vals[0])>0.74)
      reward = std::min((Real)-1.,reward); //gently push sim away from extreme curvature: not kosher
    if(std::fabs(t_a.vals[1])>0.49)
      reward = std::min((Real)-1.,reward); //gently push sim away from extreme acceleration: not kosher

    return new_sample;
}
