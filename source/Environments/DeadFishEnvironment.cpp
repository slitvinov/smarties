/*
 *  TwoFishEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2015
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "DeadFishEnvironment.h"

using namespace std;

DeadFishEnvironment::DeadFishEnvironment(const int _nAgents, const string _execpath, const int _rank, Settings & settings):
Environment(_nAgents, _execpath, _rank, settings), // The following are not useful for this environment
sight(settings.senses==0 || settings.senses==4),
POV(settings.senses==1 || settings.senses==4), l_line(settings.senses==2),
p_sensors(settings.senses==3 || settings.senses==4), study(settings.rewardType),
goalDY((settings.goalDY>1.)? 1.-settings.goalDY : settings.goalDY)
{
	cheaperThanNetwork = false; //this environment is more expensive to simulate than updating net. todo: think it over?
}

void DeadFishEnvironment::setDims()
{
    {
        sI.inUse.clear();
        {
            sI.inUse.push_back(true);
            sI.inUse.push_back(true);
            // ...second last action (HAX!)
            sI.inUse.push_back(true); //if l_line i have curvature info
            sI.inUse.push_back(true); //if l_line i have curvature info
            sI.inUse.push_back(true); //if l_line i have curvature info
            sI.inUse.push_back(true); //if l_line i have curvature info
        }
        {
            //Dist 6
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);
            sI.inUse.push_back(false);

        }

    }
    {
        aI.dim = 1;
        aI.values.resize(aI.dim);

        //accel = 4*Ltow*length/(Tperiod*Tperiod);
        // Hardcode Ltow = 1.5, Tperiod = 2. KEEP THIS CONST
        // so bounds = +/-1.5. Rescale in MRAG by multiplying Length_fisch


        for (int i=0; i<aI.dim; i++) {
            aI.values[i].push_back(-5*1.5);
            aI.values[i].push_back(+5*1.5);
        }
    }
    resetAll=true;
    commonSetup();
}

bool DeadFishEnvironment::pickReward(const State& t_sO, const Action& t_a,
                                const State& t_sN, Real& reward, const int info)
{

    /*if (fabs(t_sN.vals[5] -t_sO.vals[4])>0.001) {
        printf("Mismatch new and old state!!! \n %s \n %s \n",
                t_sO.print().c_str(),t_sN.print().c_str());
        abort();
    }
    if (fabs(t_sN.vals[4] -t_a.vals[0])>0.001) {
        printf("Mismatch state and action!!! \n %s \n %s \n",
                t_a.print().c_str(),t_sN.print().c_str());
        abort();
    }


    if ( fabs(t_sN.vals[3] -t_sO.vals[3])<1e-2 && reward>0 ) {
        printf("Same time for two states!!! \n %s \n %s \n",t_sO.print().c_str(),t_sN.print().c_str());
        abort();
    }*/
    /*
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

    //bool new_sample(false);
    //if (reward<-9.9) new_sample=true; // failuer condition NEED TO THINK ABOUT FAILURE REWARD

    /*if (study == 0) {
        //const Real scaledEfficiency = t_sN.vals[13];
        const Real scaledEfficiency = (t_sN.vals[13]-.4)/(1.-.4); //between 0 and 1
        reward = scaledEfficiency; //max cumulative reward = sum gamma^t r < 1/(1-gamma)
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
    }
    else if (study == 1) {
        const Real scaledBndEfficiency = (t_sN.vals[16]-.3)/(.6-.3); //between 0 and 1
        reward = scaledBndEfficiency;
        if (new_sample) reward = -1./(1.-gamma);
    }
    else if (study == 2) {
        //const Real scaledRew = 1. -2.*fabs(t_sN.vals[1]-goalDY);
        const Real scaledRew = 1. -2.*sqrt(fabs(t_sN.vals[1]));
        reward =  scaledRew;
        new_sample = false;
        if (new_sample) reward = -1./(1.-gamma);
    }
    else if (study == 3) {
    	const Real DX_penal = 8*fabs(t_sN.vals[0]-goalDY);  //goalDY actually goalDX
    	const Real DY_penal = 2*fabs(t_sN.vals[1]);
        const Real scaledRew = 1. - min(DX_penal+DY_penal, 2.);
        reward = scaledRew;            //max cumulative reward = sum gamma^t r < 1/(1-gamma)
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
    }
    else if (study == 4) {
        const Real scaledEfficiency = 0.;
        reward = scaledEfficiency;            //max cumulative reward = sum gamma^t r < 1/(1-gamma)
        if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
    }
    else if (study == 5) {
        reward = (t_sN.vals[16]-.3)/(.6-.3) - 2*t_sN.vals[1]*t_sN.vals[1];
    }
    else {
        die("Wrong reward\n");
    }*/

    //return new_sample;
    return info==2;
}
