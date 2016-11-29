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

CMAEnvironment::CMAEnvironment(const int nAgents, const string execpath, const int _rank, Settings & settings) :
Environment(settings.nThreads, execpath, _rank, settings)
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

		std::uniform_real_distribution<Real> dist(0.,0.1);
		if (fabs(agents[iAgent]->a->vals[0]) > .4 ||
			fabs(agents[iAgent]->a->vals[0]) <= 0   )
				 agents[iAgent]->a->vals[0] = dist(*g);

	}  else if(aI.dim > 1) {

		std::uniform_real_distribution<Real> dist(0.,0.1);
		if (fabs(agents[iAgent]->a->vals[0]) > .4 ||
			fabs(agents[iAgent]->a->vals[0]) <= 0   )
				 agents[iAgent]->a->vals[0] = dist(*g);

	}  else if(aI.dim > 2) {

		std::uniform_real_distribution<Real> dist(0.,1.);
		if (fabs(agents[iAgent]->a->vals[0]) > .9 ||
			fabs(agents[iAgent]->a->vals[0]) <= 0   )
				 agents[iAgent]->a->vals[0] = dist(*g);

	}  else if(aI.dim > 3) {

		std::uniform_real_distribution<Real> dist(0.,1.);
		if (fabs(agents[iAgent]->a->vals[0]) > .9 ||
			fabs(agents[iAgent]->a->vals[0]) <= 0   )
				 agents[iAgent]->a->vals[0] = dist(*g);

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
    }
    {
        aI.dim = 4; //number of action that agent can perform per turn: usually 1 (eg DQN)
        aI.values.resize(aI.dim);
        for (int i=0; i<aI.dim; i++) {
        	const int nOptions = 5; //used if discrete actions: options available to agent for acting
            aI.bounds.push_back(nOptions);

            //this framework sends a real number to the application
            //if you want to receive an integer number between 0 and nOptions (eg action option)
            //just write aI.values[i].push_back(0.1); ... aI.values[i].push_back((nOptions-1) + 0.1);
            //i added the 0.1 is just to be extra safe when converting a float to an integer

            aI.values[i].push_back(.01); //here the app accepts real numbers
            aI.values[i].push_back(.03);
            aI.values[i].push_back(.05);
            aI.values[i].push_back(.07);
            aI.values[i].push_back(.09);
            //the number of components must be ==nOptions
        }
    }
    commonSetup(); //required
}

bool CMAEnvironment::pickReward(const State & t_sO, const Action & t_a, const State & t_sN, Real & reward, const int info)
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

/*
/// IF YOU WANT TO RUN AN APP DIRECTLY INSIDE SMARTIES:
NEED TO

1) override these two function with empty functions to avoid calling any executable:

	void Environment::setup_Comm() {}

	void Environment::spawn_server(){ }

2) define this function (iAgent is zero if only one agent per simulation is present)

	void Environment::setAction(const int & iAgent)
	{
	 	 for (int i=0; i<aI.dim; i++) agents[iAgent]->a->valsContinuous[i]... do something

	 	 this function loads the actions from the agent and sends it to the app
	 	 if your application is local, then find a way to process the action locally
	}


3) define this function

	int Environment::getState(int & iAgent)
	{
		get the agent number, if one agent per game iAgent=0
		load the state into agents[iAgent]->s->vals[] and
    	return bStatus;

    	bStatus is 1 is first communication about the first state (no previous actions)
    	bStatus is 2 if application is sending the last state (and requires no action!!)
    	bStatus is 0 if i already sent at least one state in the past
	}

4) think about:
	- functions 2 and 3 need to handle all the work
	- RL first calls getState, then alternates setAction and getState untill terminal state
	checklist:
	- initializing the simulation
	- sending the first state without prior actions
	- catching a terminal state
	- restarting the simulation
*/
