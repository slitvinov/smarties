/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "CMAEnvironment.h"
#include <sys/types.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <cstdio>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <iostream>
#include <algorithm>
#include <stdio.h>

using namespace std;

CMAEnvironment::CMAEnvironment(const int _nAgents, const string _execpath,
															 const int _rank, Settings & settings) :
Environment(settings.nThreads, _execpath, _rank, settings)
{
}

bool CMAEnvironment::predefinedNetwork(Network* const net) const
{
	//this function can be used if environment requires particular network settings
	//i.e. not fully connected LSTM/FF network
	//i.e. if you want to use convolutions
	return false;
}



void CMAEnvironment::setAction(const int & iAgent)
{
   if(aI.dim > 0) {
		std::uniform_real_distribution<Real> dist(.01,0.1);
		if (agents[iAgent]->a->vals[0] > .2 ||
		    agents[iAgent]->a->vals[0] < .0 )
			 agents[iAgent]->a->vals[0] = dist(*g);

	}
	if(aI.dim > 1) {

		std::uniform_real_distribution<Real> dist(.01,0.1);
		if (agents[iAgent]->a->vals[1] > .2 ||
			 agents[iAgent]->a->vals[1] < .0 )
			 agents[iAgent]->a->vals[1] = dist(*g);

	}
	if(aI.dim > 2) {

		std::uniform_real_distribution<Real> dist(.1,.9);
		if (agents[iAgent]->a->vals[2] > .9 ||
		    agents[iAgent]->a->vals[2] < 0. )
			 agents[iAgent]->a->vals[2] = dist(*g);

	}
	if(aI.dim > 3) {

		std::uniform_real_distribution<Real> dist(.1,.9);
		if (agents[iAgent]->a->vals[3] > .9 ||
			  agents[iAgent]->a->vals[3] < 0. )
		    agents[iAgent]->a->vals[3] = dist(*g);

	}
	if(aI.dim > 4) {
		std::uniform_real_distribution<Real> dist(1.1,2.5);
		if (agents[iAgent]->a->vals[4]  > 5. ||
			  agents[iAgent]->a->vals[4] < 1. )
		    agents[iAgent]->a->vals[4] = dist(*g);

	}
	if(aI.dim > 5) {
		std::uniform_real_distribution<Real> dist(.5,2.);
		if (agents[iAgent]->a->vals[5] > 100. ||
			  agents[iAgent]->a->vals[5] < 0. )
		    agents[iAgent]->a->vals[5] = dist(*g);

	}  else die("No actions sent?\n");

	Environment::setAction(iAgent);
}

void CMAEnvironment::setDims() //this environment is for the cart pole test
{
    {
        sI.inUse.clear();
        //for each state variable:
        // State: ratio between min/max std...
        sI.inUse.push_back(true);//ignore, leave as is

        // ...ratio between min/max eigenvalues of covariance...
        sI.inUse.push_back(true);//ignore, leave as is

        // ...progress rate...
        sI.inUse.push_back(true); //ignore, leave as is

        // ...function change...
        sI.inUse.push_back(true); //ignore, leave as is

        // ...dimensionality...
        sI.inUse.push_back(true); //ignore, leave as is

        // ...psigma...
        sI.inUse.push_back(true); //ignore, leave as is
    }
    {
        aI.dim = 6; //number of action that agent can perform per turn: usually 1 (eg DQN)
        aI.values.resize(aI.dim);
        for (int i=0; i<2; i++) {
        	const int nOptions = 5; //used if discrete actions: options available to agent for acting
            aI.bounds.push_back(nOptions);

            aI.values[i].push_back(.01); //here the app accepts real numbers
            aI.values[i].push_back(.03);
            aI.values[i].push_back(.05);
            aI.values[i].push_back(.07);
            aI.values[i].push_back(.09);
        }
        for (int i=2; i<4; i++) {
        	const int nOptions = 5; //used if discrete actions: options available to agent for acting
            aI.bounds.push_back(nOptions);

            aI.values[i].push_back(.1); //here the app accepts real numbers
            aI.values[i].push_back(.3);
            aI.values[i].push_back(.5);
            aI.values[i].push_back(.7);
            aI.values[i].push_back(.9);
        }
            aI.bounds.push_back(3);
            aI.values[4].push_back(1.2); //here the app accepts real numbers
            aI.values[4].push_back(1.8);
            aI.values[4].push_back(2.5);

            aI.bounds.push_back(3);
            aI.values[5].push_back(.5); //here the app accepts real numbers
            aI.values[5].push_back(1.);
            aI.values[5].push_back(2.);

    }
    commonSetup(); //required
}

bool CMAEnvironment::pickReward(const State& t_sO, const Action& t_a,
															  const State& t_sN, Real& reward, const int info)
{
    bool new_sample(info == 2);

    //Compute the reward. If you do not do anything, reward will be whatever was set already to reward.
    //this means that reward will be one sent by the app

    //if (reward<-0.9) new_sample=true; //in cart pole example, if reward from the app is -1 then I failed

    //here i can change the reward: instead of -1 or 0, i can give a positive reward if angle is small
    //reward = 1. - fabs(t_sN.vals[3])/0.2;    //max cumulative reward = sum gamma^t r < 1/(1-gamma)
    //if (new_sample) reward = -1./(1.-gamma); // = - max cumulative reward
    //was is the last state of the sequence?

    //this must be set: was it the last episode? you can get it from reward?
    return new_sample; //cart pole has failed if r = -1, need to clean this shit and rely only on info
}
