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
Environment(_nAgents, _execpath, _rank, settings),
sight( settings.senses    ==0),
rcast( settings.senses    % 2), //if eq {1,  3,  5,  7}
lline((settings.senses/2) % 2), //if eq {  2,3,    6,7}
press((settings.senses/4) % 2), //if eq {      4,5,6,7}
study(settings.rewardType)
{
	cheaperThanNetwork = false; //this environment is more expensive to simulate than updating net. todo: think it over?
}

void DeadFishEnvironment::setDims()
{
		sI.inUse.clear();

		sI.inUse.push_back(sight); //relative x
		sI.inUse.push_back(sight); //argument of sinusoidal
		sI.inUse.push_back(sight); //relative y
		sI.inUse.push_back(false); //current y in sim frame, not used for NN
		sI.inUse.push_back(sight); //relative angle
		sI.inUse.push_back(false); //current angle in sim frame, not used for NN
		sI.inUse.push_back(true); //last action (acceleration)
		sI.inUse.push_back(true); //absolute vx
		sI.inUse.push_back(true); //absolute vy
		sI.inUse.push_back(true); //relative vx
		sI.inUse.push_back(true); //relative vy
		sI.inUse.push_back(false); //absolute angular vel
		sI.inUse.push_back(false); //distance
		sI.inUse.push_back(false); //quadrant

		{ //number this is c-style index
				// force x avg 14
				sI.inUse.push_back(false);

				// force y avg 15
				sI.inUse.push_back(false);

				//Pout 16
				sI.inUse.push_back(false);

				//defPower 17
				sI.inUse.push_back(false);

				// EffPDef 18
				sI.inUse.push_back(false);

				// PoutBnd 19
				sI.inUse.push_back(false);

				// defPowerBnd 20
				sI.inUse.push_back(false);

				// EffPDefBnd 21
				sI.inUse.push_back(false);

				// Pthrust 22
				sI.inUse.push_back(false);

				// Pdrag 23
				sI.inUse.push_back(false);

				// ToD 24
				sI.inUse.push_back(false);

				// single guy miles per gallon 25
				sI.inUse.push_back(false);

				// two guys miles per gallon 26
				sI.inUse.push_back(false);
		}
		const int nSensors = 20;

		//velocity sensors
		for (int i=0; i<4*nSensors; i++) sI.inUse.push_back(press);

		//force sensors
		for (int i=0; i<4*nSensors; i++) sI.inUse.push_back(press);

		//sight rays
		for (int i=0; i<2*nSensors; i++) sI.inUse.push_back(rcast);

    aI.dim = 1;
    aI.values.resize(aI.dim);

    //accel = 4*Ltow*length/(Tperiod*Tperiod);
    // Hardcode Ltow = 1.5, Tperiod = 2. KEEP THIS CONST
    // so bounds = +/-1.5. Rescale in MRAG by multiplying Length_fisch
    for (int i=0; i<aI.dim; i++) {
        aI.values[i].push_back(-5*1.5);
        aI.values[i].push_back(+5*1.5);
    }

    resetAll=false; //all agents send states upon a failure
    commonSetup();
}

bool DeadFishEnvironment::pickReward(const State& t_sO, const Action& t_a,
                                const State& t_sN, Real& reward, const int info)
{
	//here we can check that we correclt gave old action in state:
	if (fabs(t_sN.vals[6] -t_a.vals[0])>1e-10) {
			printf("Mismatch state and action!!! %s === %s\n",
			 t_sN.print().c_str(),t_a.print().c_str());
			abort();
	}

	const bool terminated = info==2;
	//assert(terminated == (reward<-9.9));

	if (study == 1) reward = t_sN.vals[25]; //then we use single guy miles per gallon

	if (terminated) reward = -2./(1.-gamma);
  return terminated;
}
