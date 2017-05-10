/*
 *  ExternalEnvironment.cpp
 *  smarties
 *
 *  Created by Guido Novati on May 13, 2016
 *  Copyright 2015 ETH Zurich. All rights reserved.
 *
 */

#include "CartEnvironment.h"
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

openAICartEnvironment::openAICartEnvironment(const int _nAgents, const string _execpath,
																 const int _rank, Settings & settings) :
Environment(_nAgents, _execpath, _rank, settings), allSenses(settings.senses==0)
{
//   cheaperThanNetwork=false;
}

bool openAICartEnvironment::predefinedNetwork(Network* const net) const
{
	//this function can be used if environment requires particular network settings
	//i.e. not fully connected LSTM/FF network
	//i.e. if you want to use convolutions
	return false;
}

void openAICartEnvironment::setDims() //this environment is for the cart pole test
{
    {
        sI.inUse.clear();
        //for each state variable:
        // State: coordinate...
        sI.inUse.push_back(true); //ignore, leave as is

        // ...velocity...
		sI.inUse.push_back(allSenses); //ignore, leave as is

        // ...and angular velocity
		sI.inUse.push_back(allSenses); //ignore, leave as is

        // ...angle...
		sI.inUse.push_back(true); //ignore, leave as is

        /*
         * also valid:
         *
         * for (int i=0; i<some_number_of_vars; i++)
         * {
         * 		sI.top.push_back(MAXVAL); sI.bottom.push_back(MINVAL);
         * 		sI.isLabel.push_back(false); sI.inUse.push_back(true); sI.bounds.push_back(1); //ignore, leave as is
         * }
         */
    }
    {
        aI.dim = 1; //number of action that agent can perform per turn: usually 1 (eg DQN)
        aI.values.resize(aI.dim);
        for (int i=0; i<aI.dim; i++) {

            //this framework sends a real number to the application
            //if you want to receive an integer number between 0 and nOptions (eg action option)
            //just write aI.values[i].push_back(0.1); ... aI.values[i].push_back((nOptions-1) + 0.1);
            //i added the 0.1 is just to be extra safe when converting a float to an integer

            aI.values[i].push_back(0.0);
            aI.values[i].push_back(1.0);
            //the number of components must be ==nOptions
        }
    }
    commonSetup(); //required
}

bool openAICartEnvironment::pickReward(const State & t_sO, const Action & t_a,
																 const State& t_sN, Real& reward,const int info)
{
    return info==2;
}
